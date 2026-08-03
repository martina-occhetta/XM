"""
Quick feasibility test: does Explorative Modeling (best-of-K exploration) help a
perturbation-response *generative* model (scLDM-style) avoid mode averaging?

Why this is the right toy problem
---------------------------------
Perturbation-response models (e.g. scLDM) are conditional latent generative
models: given a perturbation (drug / gene KO) -- and optionally a control state
-- predict the *distribution* of perturbed single-cell states. The signature
failure mode in this field is regression-to-the-mean: a perturbation drives a
**heterogeneous** population (responder / non-responder subpopulations, distinct
transcriptional programs), i.e. a multimodal target, but models trained with a
per-sample reconstruction objective and an arbitrary noise<->data pairing tend
to smear probability mass into the low-density valley *between* the response
modes. That is exactly the "each prediction averages across many modes"
pathology Explorative Modeling attacks: explore K candidate noises, train only
on the best-matching one, so each update commits to a single mode.

This script reproduces that structure in a controlled 2-D latent (so we can
measure and see mode averaging directly) and compares:
    * baseline  (xm_best_of_k = 1)  -- ordinary conditional flow matching
    * XM        (xm_best_of_k = 4, 8) -- identical model + the repo's real
      `xm_chunked_best_of_k` selection engine (vendored in xm_core.py)

The flow-matching interpolation matches the repo's convention exactly
(CondOTProbPath: x_t = (1-t)*noise + t*data, target velocity u_t = data-noise;
see model/flow/scheduler.py and model/flow/flow_matching.py).

Everything runs on CPU in a few minutes. Metrics + a figure are written to
this directory.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from xm_core import xm_chunked_best_of_k

DEVICE = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Synthetic perturbation-response dataset
# ---------------------------------------------------------------------------
class PerturbationData:
    """
    P perturbations. Each perturbation p induces a BIMODAL perturbed-state
    distribution in a 2-D latent: a mixture of two Gaussians (two response
    programs) with mixture weight w_p. Mode centers are spread around a ring so
    perturbations are distinguishable. The two modes of a perturbation are placed
    on opposite sides of that perturbation's mean shift, so the *conditional mean*
    (what a mode-averaging model collapses to) lands in the empty valley between
    them -- making mode averaging measurable.
    """

    def __init__(self, n_perturbations=8, mode_sigma=0.16, gap=2.2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.P = n_perturbations
        self.mode_sigma = mode_sigma
        angles = torch.linspace(0, 2 * math.pi, n_perturbations + 1)[:-1]
        # per-perturbation base location (the "mean shift" direction), on a ring
        base = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 2.5
        # split direction for the two modes (tangent to the ring), scaled by gap
        split = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=1) * (gap / 2)
        self.mode0 = base + split           # (P, 2)
        self.mode1 = base - split           # (P, 2)
        # mixture weights per perturbation in [0.35, 0.65] (imbalanced responses)
        self.w0 = 0.35 + 0.30 * torch.rand(n_perturbations, generator=g)

    def sample(self, n_per_pert, seed=None):
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        xs, conds = [], []
        for p in range(self.P):
            n0 = int(round(n_per_pert * self.w0[p].item()))
            n1 = n_per_pert - n0
            for n, center in ((n0, self.mode0[p]), (n1, self.mode1[p])):
                if n == 0:
                    continue
                noise = torch.randn(n, 2, generator=g) * self.mode_sigma
                xs.append(center.unsqueeze(0) + noise)
                conds.append(torch.full((n,), p, dtype=torch.long))
        x = torch.cat(xs, 0)
        c = torch.cat(conds, 0)
        perm = torch.randperm(x.shape[0], generator=g)
        return x[perm], c[perm]

    def true_samples_for(self, p, n, seed):
        g = torch.Generator().manual_seed(seed)
        n0 = int(round(n * self.w0[p].item()))
        n1 = n - n0
        s0 = self.mode0[p].unsqueeze(0) + torch.randn(n0, 2, generator=g) * self.mode_sigma
        s1 = self.mode1[p].unsqueeze(0) + torch.randn(n1, 2, generator=g) * self.mode_sigma
        return torch.cat([s0, s1], 0)


# ---------------------------------------------------------------------------
# Conditional velocity field  v_theta(x_t, t, perturbation)
# ---------------------------------------------------------------------------
class CondVelocity(nn.Module):
    def __init__(self, dim=2, n_perturbations=8, embed=32, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(n_perturbations, embed)
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + embed, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t, t, cond):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        h = torch.cat([x_t, t, self.embed(cond)], dim=-1)
        return self.net(h)


# ---------------------------------------------------------------------------
# Flow-matching loss wrapper compatible with xm_chunked_best_of_k.
# Exploration is over the noise (rand_inputs); t is held fixed per sample and
# passed inside `conditions` -- exactly like model/img/dit_cc.py.
# ---------------------------------------------------------------------------
def loss_calc_wrapper(model_forward, conditions, gt_samples, learning=True,
                      rand_inputs=None, rand_seeds=None):
    t, cond = conditions                      # t: (B,1), cond: (B,)
    with torch.set_grad_enabled(learning):
        x0 = rand_inputs                       # noise
        x1 = gt_samples                        # data (perturbed state)
        x_t = (1.0 - t) * x0 + t * x1          # CondOTProbPath interpolation
        u_t = x1 - x0                          # target velocity
        v = model_forward(x_t, t.squeeze(-1), cond)
        per_sample = (v - u_t).pow(2).reshape(x1.shape[0], -1).mean(dim=1)
        return per_sample, None                # (losses, predictions)


@torch.no_grad()
def generate(model, cond, n_steps=8):
    """Euler ODE integration from noise (t=0) to data (t=1)."""
    x = torch.randn(cond.shape[0], 2, device=DEVICE)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((cond.shape[0],), i * dt, device=DEVICE)
        x = x + dt * model(x, t, cond)
    return x


def train(best_of_k, data, seed, updates=4000, batch=256, lr=2e-3):
    torch.manual_seed(seed)
    model = CondVelocity(n_perturbations=data.P).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, C = data.sample(4000, seed=1000 + seed)       # training pool (shared per seed)
    X, C = X.to(DEVICE), C.to(DEVICE)
    N = X.shape[0]
    model.train()
    for step in range(updates):
        idx = torch.randint(0, N, (batch,), device=DEVICE)
        x1, cond = X[idx], C[idx]
        t = torch.rand(batch, 1, device=DEVICE)      # fixed per sample (explored over noise only)
        opt.zero_grad()
        losses, _ = xm_chunked_best_of_k(
            model.forward, loss_calc_wrapper,
            conditions=(t, cond), gt_samples=x1,
            best_of_k=best_of_k, max_chunk_bs_mult=max(best_of_k, 1),
            save_mem_mode=True, not_training=False,
        )
        losses.mean().backward()
        opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def energy_distance(x, y):
    """Székely energy distance between two point clouds (lower = closer)."""
    def pdist_mean(a, b):
        return torch.cdist(a, b).mean()
    return (2 * pdist_mean(x, y) - pdist_mean(x, x) - pdist_mean(y, y)).item()


def gap_occupancy(gen, c0, c1, band=0.5, lo=0.30, hi=0.70):
    """
    Fraction of generated points that fall in the *valley between* the two true
    modes -- the direct signature of mode averaging. A point is "in the gap" if
    its projection onto the segment c0->c1 lands in [lo, hi] and it is within
    `band` (units of inter-mode distance) of the connecting line.
    """
    d = c1 - c0
    L2 = (d * d).sum()
    rel = gen - c0
    s = (rel @ d) / L2                          # projection parameter along c0->c1
    proj = c0 + s.unsqueeze(-1) * d
    perp = (gen - proj).norm(dim=-1) / d.norm()  # perpendicular dist (rel. units)
    in_gap = (s >= lo) & (s <= hi) & (perp <= band)
    return in_gap.float().mean().item()


def mode_recall(gen, c0, c1):
    """Both modes covered? Returns min mass assigned to either mode (0..0.5)."""
    d0 = (gen - c0).norm(dim=-1)
    d1 = (gen - c1).norm(dim=-1)
    assign1 = (d1 < d0).float().mean().item()
    return min(assign1, 1 - assign1)


def evaluate(model, data, n_steps=8, n_eval=2000):
    eds, gaps, recalls = [], [], []
    with torch.no_grad():
        for p in range(data.P):
            cond = torch.full((n_eval,), p, dtype=torch.long, device=DEVICE)
            gen = generate(model, cond, n_steps=n_steps)
            true = data.true_samples_for(p, n_eval, seed=7777 + p).to(DEVICE)
            eds.append(energy_distance(gen, true))
            gaps.append(gap_occupancy(gen, data.mode0[p], data.mode1[p]))
            recalls.append(mode_recall(gen, data.mode0[p], data.mode1[p]))
    # true-data gap occupancy as a floor reference (should be ~0)
    true_gap = []
    for p in range(data.P):
        true = data.true_samples_for(p, n_eval, seed=13 + p).to(DEVICE)
        true_gap.append(gap_occupancy(true, data.mode0[p], data.mode1[p]))
    return {
        "energy_distance": float(np.mean(eds)),
        "gap_occupancy": float(np.mean(gaps)),
        "mode_recall": float(np.mean(recalls)),
        "true_data_gap_occupancy": float(np.mean(true_gap)),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    data = PerturbationData(seed=0)
    results = {}
    per_seed = {}
    for k in args.ks:
        metrics_over_seeds = []
        for s in args.seeds:
            model = train(k, data, seed=s, updates=args.updates)
            m = evaluate(model, data, n_steps=args.n_steps)
            metrics_over_seeds.append(m)
            print(f"[K={k} seed={s}] " + " ".join(f"{kk}={vv:.4f}" for kk, vv in m.items()))
        # aggregate mean/std
        agg = {}
        for key in metrics_over_seeds[0]:
            vals = [mm[key] for mm in metrics_over_seeds]
            agg[key + "_mean"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        results[f"K={k}"] = agg
        per_seed[f"K={k}"] = metrics_over_seeds
        print(f"==> K={k} AGG: gap={agg['gap_occupancy_mean']:.4f}+-{agg['gap_occupancy_std']:.4f} "
              f"ED={agg['energy_distance_mean']:.4f}+-{agg['energy_distance_std']:.4f} "
              f"recall={agg['mode_recall_mean']:.4f}\n")

    # compute-matched baseline: K=1 with the largest-K's worth of extra updates
    if 1 in args.ks and max(args.ks) > 1:
        kmax = max(args.ks)
        cm = []
        for s in args.seeds:
            model = train(1, data, seed=s, updates=args.updates * kmax)
            cm.append(evaluate(model, data, n_steps=args.n_steps))
            print(f"[K=1 x{kmax} updates seed={s}] gap={cm[-1]['gap_occupancy']:.4f} ED={cm[-1]['energy_distance']:.4f}")
        agg = {}
        for key in cm[0]:
            agg[key + "_mean"] = float(np.mean([mm[key] for mm in cm]))
            agg[key + "_std"] = float(np.std([mm[key] for mm in cm]))
        results[f"K=1_x{kmax}updates"] = agg
        per_seed[f"K=1_x{kmax}updates"] = cm

    out = {"config": vars(args), "results": results, "per_seed": per_seed}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")

    if args.plot:
        make_plot(data, args)


def make_plot(data, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [k for k in args.ks]
    seed = args.seeds[0]
    models = {k: train(k, data, seed=seed, updates=args.updates) for k in ks}
    p_show = list(range(min(4, data.P)))
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(4 * (len(ks) + 1), 4))
    colors = plt.cm.tab10.colors

    # panel 0: ground truth
    ax = axes[0]
    for p in p_show:
        true = data.true_samples_for(p, 800, seed=42 + p)
        ax.scatter(true[:, 0], true[:, 1], s=3, color=colors[p], alpha=0.4)
    ax.set_title("Ground truth\n(bimodal per perturbation)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    for j, k in enumerate(ks):
        ax = axes[j + 1]
        for p in p_show:
            cond = torch.full((800,), p, dtype=torch.long)
            gen = generate(models[k], cond, n_steps=args.n_steps).cpu()
            ax.scatter(gen[:, 0], gen[:, 1], s=3, color=colors[p], alpha=0.4)
        label = "baseline (no exploration)" if k == 1 else f"XM best-of-{k}"
        ax.set_title(f"{label}\n(K={k})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Explorative Modeling on a perturbation-response toy "
                 f"({args.n_steps}-step generation) — does exploration avoid the between-mode valley?",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(HERE, "perturbation_xm_samples.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
