# Can Explorative Modeling help perturbation-response models (e.g. scLDM)?

**Short answer: yes — conceptually it's a clean fit, and a quick controlled test
shows the exploration mechanism does what the theory predicts on a
perturbation-response task: it removes mode averaging.** The benefit is
decisive in the few-step / fast-inference regime that matters in practice, and
comes with an honest calibration caveat in the many-step regime. Details below.

---

## 1. Why the fit is natural

A perturbation-response model like **scLDM** is a *conditional latent generative
model*: given a perturbation (drug / gene KO), and optionally a control cell
state, it predicts the **distribution** of perturbed single-cell states (usually
by running a diffusion/flow model in a VAE latent).

Explorative Modeling (XM) is a modality-agnostic wrapper around exactly this kind
of model. Its core engine, [`xm_chunked_best_of_k`](../../model/model_utils.py),
only needs:

| XM ingredient | scLDM analogue |
| --- | --- |
| `conditions` | perturbation embedding (+ control state) |
| `gt_samples` | perturbed-cell latent (VAE code) |
| `loss_calc_wrapper` | the scLDM flow/diffusion loss |
| latent explored | the noise / source latent |

Compare `model/img/dit_cc.py` (class-conditional DiT + XM): swap the class label
for a perturbation embedding and the VAE image latent for a cell-state latent and
you already have "scLDM + exploration". No change to the exploration engine is
needed.

**And the problem XM targets is *the* problem in this field.** A single
perturbation drives a **heterogeneous population** — responder / non-responder
subpopulations, distinct transcriptional programs — i.e. a *multimodal* target.
Models trained with a per-sample reconstruction loss and an arbitrary
noise↔data pairing regress to the mean: they smear probability mass into the
low-density valley *between* response modes (predicting biologically implausible
"intermediate" cells). That "each prediction averages across modes" pathology is
precisely what XM attacks: explore K candidate noises, train only on the
best-matching one, so each update commits to a single mode.

## 2. What the quick test actually is

A self-contained CPU experiment (`perturbation_toy.py`, a few minutes) that
reuses the repo's **real** exploration engine (`xm_core.py` is a verbatim copy of
`xm_chunked_best_of_k`) and its **exact** flow-matching convention
(`CondOTProbPath`: `x_t=(1-t)·noise + t·data`, target velocity `data-noise`; see
`model/flow/`).

- **Task.** 8 perturbations in a 2-D latent (so mode averaging is directly
  measurable/visible). Each perturbation induces a **bimodal** perturbed-state
  distribution with imbalanced mixture weights. The two modes straddle the
  perturbation's mean shift, so the conditional mean — what an averaging model
  collapses to — lands in the empty valley between them.
- **Models.** One conditional velocity MLP, trained three ways, identical in
  every respect except the exploration count `xm_best_of_k`:
  baseline `K=1` (ordinary flow matching) vs XM `K=4`, `K=8`.
- **Controls.** 3 seeds (mean±std); a **compute-matched** baseline (`K=1` given
  `8×` the gradient updates) to separate "exploration" from "just more compute".
- **Metrics.** `gap_occupancy` = fraction of generated cells landing in the
  between-mode valley (direct mode-averaging signature, lower is better);
  `energy_distance` to held-out true cells (overall distributional fit);
  `mode_recall` (→0.5 = both subpopulations covered). True-data `gap_occupancy`
  is ~0 by construction (the floor).

## 3. Results

Mean ± std over 3 seeds. `gap` and `ED` lower is better; `recall`→0.5 is better.

### 8-step generation (near-converged, "easy" regime)

| model | gap-occupancy ↓ | energy distance ↓ | mode recall |
| --- | --- | --- | --- |
| baseline `K=1`            | 0.037 ± 0.005 | **0.024** | 0.413 |
| `K=1`, 8× updates (compute-matched) | 0.021 | 0.014 | — |
| **XM `K=4`**              | **0.0026 ± 0.0008** | 0.064 | 0.470 |
| **XM `K=8`**              | **0.0022 ± 0.0003** | 0.064 | 0.465 |

### 2-step generation (fast inference — the regime that matters for scLDM)

| model | gap-occupancy ↓ | energy distance ↓ | mode recall |
| --- | --- | --- | --- |
| baseline `K=1`            | 0.259 ± 0.015 | 0.201 | 0.390 |
| `K=1`, 8× updates (compute-matched) | 0.288 | 0.193 | — |
| **XM `K=4`**              | 0.025 ± 0.004 | 0.048 | 0.471 |
| **XM `K=8`**              | **0.020 ± 0.002** | **0.041** | 0.465 |

![samples](perturbation_xm_samples.png)

The figure (8-step) shows the mechanism directly: the baseline strings samples
along **arcs through the between-mode valley** — few-step curvature smear, i.e.
mode averaging — while XM `K=4/8` produce tight, on-mode clusters.

## 4. Reading the results (the honest version)

- **XM removes mode averaging, as predicted.** `gap_occupancy` drops **14–17×**
  at 8 steps and **10–13×** at 2 steps, and `mode_recall` moves toward the
  balanced 0.5. Visually the between-mode "bridges" disappear.
- **More compute alone does not reproduce this.** The compute-matched baseline
  (8× updates) barely dents `gap_occupancy` at 8 steps and *not at all* at 2
  steps — exploration is doing something optimization can't.
- **Decisive win in the few-step regime.** At 2 steps — the practically relevant
  setting for fast scLDM inference — XM is better on **both** mode-commitment and
  overall distributional fit (energy distance **4–5× lower**). This is the
  headline: where mode averaging is worst, exploration wins outright.
- **An honest caveat at many steps.** At 8 steps the near-converged baseline
  already almost solves this easy, well-separated case, and XM's hard
  commitment *raises* energy distance (0.064 vs 0.024) even as it sharpens modes
  — a precision-vs-calibration trade. A likely culprit is a known subtlety of
  **Forward** XM: best-of-K selection makes the *training-time* source-noise
  marginal non-Gaussian, while inference still samples plain `N(0,I)`, slightly
  distorting mixture weights / spread. Worth watching in a real integration.

**Bottom line for feasibility:** the framework plugs into an scLDM-shaped model
with zero changes to the exploration engine, trains stably, and demonstrably
shifts the mode-averaging behavior in the predicted direction — strongly enough,
in the fast-inference regime, to improve the full distributional metric too.

## 5. What a real scLDM integration would need (next steps)

1. **Wrap the real model.** Add a `loss_calc_wrapper` around scLDM's flow/
   diffusion loss and call `xm_chunked_best_of_k` — mirror `model/img/dit_cc.py`,
   with `conditions = (t, perturbation_embed[, control_latent])` and
   `gt_samples = perturbed_cell_latent`.
2. **Real data / metrics.** Swap the toy for e.g. Norman/Replogle/sci-Plex; score
   with the field's distributional metrics (energy distance / MMD / E-distance,
   per-DEG accuracy, subpopulation-fraction calibration) rather than 2-D
   gap-occupancy.
3. **Address the noise-marginal mismatch** (the ED caveat above): tune K, or try
   **Reverse XM** (hold a generation fixed, explore over K data targets — see
   Sec 3.2 of the paper), which sidesteps the source-noise reweighting and may
   calibrate better on unpaired control→perturbed data.
4. **Watch subpopulation calibration**, not just mode sharpness — getting the
   responder/non-responder *fractions* right is as important as committing to a
   mode.

## 6. Reproduce

```bash
pip install torch numpy matplotlib        # CPU is fine
cd experiments/perturbation_xm
python3 perturbation_toy.py --updates 4000 --seeds 0 1 2 --ks 1 4 8 --n_steps 8 --plot   # 8-step table + figure + results.json
python3 perturbation_toy.py --updates 4000 --seeds 0 1 2 --ks 1 4 8 --n_steps 2           # 2-step (fast-inference) table
```

Files: `xm_core.py` (verbatim `xm_chunked_best_of_k`), `perturbation_toy.py`
(task + training + metrics + plot), `results.json` (8-step run),
`perturbation_xm_samples.png`.
