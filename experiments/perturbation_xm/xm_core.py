"""
xm_core.py
==========

`xm_chunked_best_of_k` is the modality-agnostic exploration engine at the heart
of Explorative Modeling. It is copied **verbatim** from
``model/model_utils.py`` (the function of the same name) so that this quick
perturbation-response feasibility test can reuse the *exact* selection logic
without importing the full training stack (``model.model_utils`` pulls in
pytorch_lightning / diffusers / torchvision / the flow_matching lib, none of
which are needed just to demonstrate the mechanism).

The only dependencies of the vendored function are ``torch`` and ``math``.
If the upstream implementation changes, re-copy it here.
"""

import math
import torch


# ---------------------------------------------------------------------------
# VERBATIM COPY of model/model_utils.py::xm_chunked_best_of_k
# ---------------------------------------------------------------------------
def xm_chunked_best_of_k(model_forward, loss_calc_wrapper, conditions, gt_samples, best_of_k, max_chunk_bs_mult, save_mem_mode = True, debug_save_mem_mode = False, not_training = False, **loss_calc_kwargs):
    """
    Parallelizes Explorative Modeling over chunks to find the best-of-k loss mode.

    This function explores multiple potential solutions in parallel (chunks) and selects the one with the minimum loss.
    Chunks allow simulating multiple batches at once for parallelization. Usually just returns losses but can also return predictions

    NOTE determinism contract: loss_calc_wrapper must be a deterministic function of
    (conditions, gt_samples, rand_inputs, rand_seeds, **loss_calc_kwargs) -- all stochasticity
    (cfg dropout masks, jump indices, per-position randomness, etc) must be derived from these
    arguments and passed in, never sampled inside the wrapper from the global RNG. save_mem_mode
    relies on this: it re-runs the wrapper with the winning rand_inputs/rand_seeds to rebuild the
    autograd graph, so any unseeded randomness means the recomputed loss silently belongs to a
    different candidate than the one selected (xm_debug_mode catches this).

    NOTE CFG/dropped-label nuance: parallelized exploration + save_mem_mode complicates CFG dropout a tad.
    The drop decision must be (1) sampled once per original sample (shape regular_bs) outside the wrapper
    and passed in via kwargs (e.g. cfg_drop_mask), then tiled across the K candidates -- so all candidates
    for a sample share the same drop regime and min-selection can't dodge dropout (dropped candidates would
    always lose to non-dropped ones on loss) -- and (2) applied identically regardless of `learning`, so the
    no-grad exploration selects noise under the exact regime the final recompute trains with.
    """

    # short-circuit for best_of_k=1: skip chunking overhead
    if best_of_k == 1:
        learning_direct = not not_training
        rand_inputs = torch.randn_like(gt_samples)
        rand_seeds = torch.randint(0, 2147483647, (gt_samples.shape[0],), device=gt_samples.device)
        losses, predictions = loss_calc_wrapper(model_forward, conditions, gt_samples, learning=learning_direct, rand_inputs=rand_inputs, rand_seeds=rand_seeds, **loss_calc_kwargs)
        return losses, predictions

    # prepare chunking variables ------------------------------------
    if debug_save_mem_mode:
        assert save_mem_mode, "debug_mode can only be used when debug_save_mem_mode is set"
    if not save_mem_mode and not not_training:
        assert max_chunk_bs_mult >= best_of_k, "save_mem_mode=False requires max_chunk_bs_mult >= best_of_k so all candidates fit in a single chunk (otherwise in-place updates to best_losses corrupt the autograd graph). use save_mem_mode=True or increase max_chunk_bs_mult"
    regular_bs = gt_samples.shape[0] # this is the regular batch sizes used for training models; we refer to this as B
    assert max_chunk_bs_mult >= 1, "need to be exploring with a max_chunk_bs_mult >= 1"
    assert best_of_k >=1, "best_of_k needs to be >= 1 for this to work"
    first_iter = True
    total_exploration_bs = regular_bs * best_of_k
    max_chunk_bs = max_chunk_bs_mult * regular_bs
    for_loop_iters = math.ceil(total_exploration_bs / max_chunk_bs)
    remaining_bs = total_exploration_bs
    learning = not save_mem_mode if not not_training else False
    best_predictions = None

    with torch.set_grad_enabled(learning):
        for _ in range(for_loop_iters):
            # prepare per iteration chunking ------------------------------------
            curr_chunk_bs = remaining_bs if remaining_bs <= max_chunk_bs else max_chunk_bs
            assert curr_chunk_bs % regular_bs == 0, "need to use a chunk thats divisible by reg bs, error occurred, please investigate"
            remaining_bs = remaining_bs - curr_chunk_bs
            this_chunk_bs_mult = int(curr_chunk_bs / regular_bs) # we refer to this_chunk_bs_mult as C_BS

            # prepare random conditions ------------------------------------
            rand_inputs = torch.randn((curr_chunk_bs, *gt_samples.shape[1:]), device=gt_samples.device) # C_BS, *gt_shape
            rand_seeds = torch.randint(0, 2147483647, (curr_chunk_bs,), device=gt_samples.device) # C_BS; we use the max 32 bit int value here

            # prepare conditions and ground truth ------------------------------------
            if isinstance(conditions, tuple):
                conditions_expanded = tuple(torch.cat([c] * this_chunk_bs_mult, dim=0) for c in conditions)
            else:
                conditions_expanded = torch.cat([conditions] * this_chunk_bs_mult, dim=0)
            gt_samples_expanded = torch.cat([gt_samples] * this_chunk_bs_mult, dim=0) # C_BS, *gt_shape

            # compute losses and possibly predictions ------------------------------------
            losses, predictions = loss_calc_wrapper(model_forward, conditions_expanded, gt_samples_expanded, learning=learning, rand_inputs=rand_inputs, rand_seeds=rand_seeds, **loss_calc_kwargs)

            # do best of k along chunk ------------------------------------
            chunk_losses_reshaped = losses.reshape(this_chunk_bs_mult, regular_bs) # C_BS, B
            chunk_min_losses, chunk_min_indices = chunk_losses_reshaped.min(dim=0) # (B,), (B,)

            # Convert chunk_min_indices (0..M-1) to flat indices (0..M*B-1), select the best candidate from the current chunk for each batch element
            flat_indices = chunk_min_indices * regular_bs + torch.arange(regular_bs, device=gt_samples.device)

            chunk_best_rand_inputs = rand_inputs[flat_indices]
            chunk_best_rand_seeds = rand_seeds[flat_indices]

            save_best_predictions = (debug_save_mem_mode and predictions is not None) if save_mem_mode else (predictions is not None)

            if save_best_predictions:
                chunk_best_predictions = predictions[flat_indices]

            if first_iter:
                first_iter = False
                best_rand_inputs = chunk_best_rand_inputs # B, *gt_shape
                best_rand_seeds = chunk_best_rand_seeds # B,
                best_losses = chunk_min_losses # B,
                if save_best_predictions:
                    best_predictions = chunk_best_predictions # B, *gt_shape
            else:
                # Update global bests if current chunk found better solutions
                replacement_mask = chunk_min_losses < best_losses
                if replacement_mask.any():
                    best_rand_inputs[replacement_mask] = chunk_best_rand_inputs[replacement_mask] # B, *gt_shape
                    best_rand_seeds[replacement_mask] = chunk_best_rand_seeds[replacement_mask] # B,
                    best_losses[replacement_mask] = chunk_min_losses[replacement_mask] # B,
                    if save_best_predictions:
                        best_predictions[replacement_mask] = chunk_best_predictions[replacement_mask] # B, *gt_shape

    # finished for loop, if save_mem_mode do last forward, else return best ------------------------------------
    if save_mem_mode:
        learning = True if not not_training else False
        torch.clear_autocast_cache() # clear cached bf16 parameter copies from the no-grad exploration loop so the recomputation builds a fresh autograd graph
        final_losses, final_predictions = loss_calc_wrapper(model_forward, conditions, gt_samples, learning=learning, rand_inputs=best_rand_inputs, rand_seeds=best_rand_seeds, **loss_calc_kwargs) # set learning to True

        if debug_save_mem_mode:
            if best_predictions is not None:
                assert torch.allclose(best_predictions, final_predictions, rtol=1e-5, atol=1e-8), "predictions did not reproduce when doing 2nd round for comp graph"
            assert torch.allclose(best_losses, final_losses, rtol=1e-5, atol=1e-8), "losses did not reproduce when doing 2nd round for comp graph"

        return final_losses, final_predictions

    else: # already computed best_losses and best_predictions
        return best_losses, best_predictions
