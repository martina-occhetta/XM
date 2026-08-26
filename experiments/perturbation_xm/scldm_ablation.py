"""
K / sampling-step ablation with bootstrap confidence intervals for XM on the
perturbation-response task, scored with the bio-perturbations E-distance
(distribution / heterogeneity fidelity -- the metric XM is expected to move).

Adds paper-grade rigor to `scldm_bio_eval.py`:
  * sweeps exploration K in {1,2,4,8} x sampling steps in {2,4,10}
  * multiple seeds (independent training runs)
  * bootstrap 95% CIs over (seed x perturbation) units, and a PAIRED bootstrap CI
    on the XM-vs-baseline improvement at matched steps (significant iff CI > 0)
  * stronger references: Identity, MeanShift, and scGen-style latent vector
    arithmetic (latent_shift)

Efficiency: each model is trained ONCE per (seed, K) and then evaluated at every
step count (sampling steps don't affect training).

Runs on synthetic data by default (real-dataset hosts are egress-blocked here;
see scldm_bio_eval.py for the real `--dataset` path -- all flags are forwarded).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from scldm_bio_eval import (
    get_data, build_space, fit_flow, flow_predictions, baseline_predictions,
    latent_shift_predictions, per_perturbation_energy, _dense,
)
from bio_perturbations.evaluator import BenchmarkEvaluator
from bio_perturbations.baselines import IdentityBaseline, MeanShiftBaseline

HERE = os.path.dirname(os.path.abspath(__file__))


def bootstrap_ci(vals, n=4000, conf=0.95, seed=0):
    """Percentile bootstrap CI of the mean of `vals`."""
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(vals, size=(n, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return {"mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(vals.size)}


def paired_improvement_ci(base_map, xm_map, n=4000, conf=0.95, seed=0):
    """Bootstrap CI of mean(base - xm) over shared (seed,pert) keys.

    Positive => XM lowers E-distance. CI excluding 0 => significant at `conf`.
    """
    keys = [k for k in base_map if k in xm_map]
    diffs = np.array([base_map[k] - xm_map[k] for k in keys], dtype=float)
    res = bootstrap_ci(diffs, n=n, conf=conf, seed=seed)
    res["fraction_pairs_improved"] = float(np.mean(diffs > 0)) if diffs.size else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--space", default="vae", choices=["logexpr", "vae", "nbvae"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 4, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--n_gen", type=int, default=200)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--vae-epochs", type=int, default=60)
    # real-data prep (forwarded to get_data; unused for synthetic)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--max-perts", type=int, default=20)
    ap.add_argument("--max-cells-per-cond", type=int, default=400)
    ap.add_argument("--min-cells-per-pert", type=int, default=0)
    ap.add_argument("--n-pseudoreplicates", type=int, default=2)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--strict-dataset", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--xm-direction", default="forward", choices=["forward", "reverse"],
                    help="forward = explore noises (default); reverse = explore data targets")
    ap.add_argument("--generator", default="flow", choices=["flow", "diffusion"],
                    help="flow = flow-matching (default); diffusion = latent DDPM (scLDM formulation)")
    ap.add_argument("--out-dir", default=HERE, help="where to write the json/png (default: script dir)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_ref, truth, source_note = get_data(args)
    genes = list(train_ref.var_names)
    conditions = list(pd.unique(train_ref.obs["condition"]))
    perts = [c for c in conditions if c != "control"]
    print(f"[data] {source_note}: {train_ref.n_obs} train / {truth.n_obs} truth, "
          f"{len(genes)} genes, {len(perts)} perts")

    space = build_space(args.space, args.latent_dim, args.vae_epochs, seed=0)
    space.fit(_dense(train_ref.layers.get("counts", train_ref.X)))
    Z = space.encode(_dense(train_ref.layers.get("counts", train_ref.X)))
    cond_ids = np.array([conditions.index(c) for c in train_ref.obs["condition"]])
    n_pca = int(min(20, len(genes) - 1, train_ref.n_obs - 1))
    evaluator = BenchmarkEvaluator.from_anndata(train_ref, n_pca_components=n_pca, min_matched_samples=2)
    print(f"[space] {space.name} (flow dim {space.dim})")

    # --- reference baselines: per-perturbation E-distance ---
    baselines = {}
    for name, pred in [
        ("identity", baseline_predictions(IdentityBaseline(), train_ref, perts)),
        ("mean_shift", baseline_predictions(MeanShiftBaseline(), train_ref, perts)),
        ("scGen_latent_shift", latent_shift_predictions(space, train_ref, conditions, "control", genes, args.n_gen)),
    ]:
        e = per_perturbation_energy(evaluator.evaluate_anndata(pred, truth))
        baselines[name] = {"per_pert": e, **bootstrap_ci(list(e.values()))}
        print(f"[baseline {name}] E-distance {baselines[name]['mean']:.3f} "
              f"[{baselines[name]['ci_low']:.3f}, {baselines[name]['ci_high']:.3f}]")

    # --- ablation grid: train once per (seed, K), evaluate at each step count ---
    # energy[K][steps] = {(seed, pert): E-distance}
    energy = {k: {s: {} for s in args.steps} for k in args.ks}
    for seed in args.seeds:
        for k in args.ks:
            model = fit_flow(Z, cond_ids, len(conditions), best_of_k=k, seed=seed,
                             updates=args.updates, direction=args.xm_direction,
                             generator=args.generator)
            for steps in args.steps:
                pred = flow_predictions(model, space, conditions, "control", genes, args.n_gen, steps,
                                        generator=args.generator)
                e = per_perturbation_energy(evaluator.evaluate_anndata(pred, truth))
                for p, val in e.items():
                    energy[k][steps][(seed, p)] = val
            print(f"  trained+eval seed={seed} K={k}")

    # --- aggregate + CIs ---
    grid = {}
    for k in args.ks:
        grid[k] = {}
        for steps in args.steps:
            vals = list(energy[k][steps].values())
            cell = bootstrap_ci(vals, seed=1000 + k * 10 + steps)
            # paired improvement vs K=1 at the same step count
            if k != 1 and 1 in args.ks:
                cell["paired_vs_K1"] = paired_improvement_ci(
                    energy[1][steps], energy[k][steps], seed=2000 + k * 10 + steps)
            grid[k][steps] = cell

    # --- report ---
    print("\n=== E-distance grid (mean [95% CI]); paired improvement vs K=1 ===")
    for steps in args.steps:
        print(f"\n-- {steps}-step generation --")
        for k in args.ks:
            c = grid[k][steps]
            line = f"  K={k}: {c['mean']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
            if "paired_vs_K1" in c:
                p = c["paired_vs_K1"]
                sig = "SIG" if p["ci_low"] > 0 else "ns"
                line += (f"  | ΔvsK1={p['mean']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] "
                         f"({sig}, {100*p['fraction_pairs_improved']:.0f}% pairs improved)")
            print(line)

    out = {
        "config": vars(args), "source": source_note, "space": space.name,
        "baselines": {n: {"mean": b["mean"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]}
                      for n, b in baselines.items()},
        "grid": {str(k): {str(s): grid[k][s] for s in args.steps} for k in args.ks},
    }
    dsuf = "" if args.xm_direction == "forward" else f"_{args.xm_direction}"
    gsuf = "" if args.generator == "flow" else f"_{args.generator}"
    out_path = os.path.join(args.out_dir, f"scldm_ablation_{args.dataset}_{space.name}{gsuf}{dsuf}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", out_path)

    if args.plot:
        make_plot(args, grid, baselines, out_path.replace(".json", ".png"))


def make_plot(args, grid, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = args.ks
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(args.steps)))
    xs = np.arange(len(ks))
    for si, steps in enumerate(args.steps):
        means = [grid[k][steps]["mean"] for k in ks]
        los = [grid[k][steps]["ci_low"] for k in ks]
        his = [grid[k][steps]["ci_high"] for k in ks]
        ax.plot(xs, means, "-o", color=colors[si], label=f"{steps}-step")
        ax.fill_between(xs, los, his, color=colors[si], alpha=0.18)
    bl_styles = {"identity": ":", "mean_shift": "--", "scGen_latent_shift": "-."}
    for name, b in baselines.items():
        ax.axhline(b["mean"], ls=bl_styles.get(name, ":"), color="gray", lw=1.2, alpha=0.9)
        ax.text(len(ks) - 1, b["mean"], f" {name}", va="center", fontsize=8, color="gray")
    ax.set_xticks(xs); ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_xlabel("Exploration budget (best-of-K);  K=1 = no exploration (baseline)")
    ax.set_ylabel("E-distance to held-out perturbed cells  (lower = better)")
    ax.set_title(f"XM K/step ablation ({args.space} latent, {len(args.seeds)} seeds, 95% CI)\n"
                 "lower E-distance = better population/heterogeneity fidelity")
    ax.legend(title="sampling steps", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
