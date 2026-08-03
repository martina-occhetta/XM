"""
End-to-end integration: Explorative Modeling (best-of-K exploration) applied to
an scLDM-shaped conditional generative model, scored on *biological* signal with
the user's `bio-perturbations` evaluator (DEG recovery, distribution fit,
PCC-Δ), against that framework's own Identity / MeanShift baselines.

What this demonstrates
----------------------
1. The XM engine (`xm_core.xm_chunked_best_of_k`, a verbatim copy of the repo's
   real function) plugs into a perturbation-response generative model with no
   changes -- exactly as it would wrap scLDM's flow/diffusion loss.
2. The model's samples pass cleanly through the bio-perturbations prediction I/O
   contract and biological evaluator.
3. Whether exploration improves *biological* metrics -- specifically the
   distribution-level fidelity (response heterogeneity) that mean-based
   benchmarks miss and that bio-perturbations is built to measure.

"mini-scLDM" vs real scLDM
--------------------------
This uses a conditional flow-matching velocity net over (standardised log1p)
expression as a compact stand-in for scLDM's latent diffusion. A real scLDM run
is the same code with (a) a trained VAE latent instead of log1p expression and
(b) scLDM's own network -- the XM wrapper and the bio evaluation are unchanged.

Data
----
Real Perturb-seq loaders (Norman/Replogle/Adamson via pertpy) live in
`bio_perturbations.datasets`; their download hosts are egress-blocked in this
sandbox, so this script generates a *realistic synthetic Perturb-seq* instead:
raw Poisson counts, several gene-KO perturbations, each with a HETEROGENEOUS
responder / non-responder split (incomplete penetrance) that creates genuine
DEGs *and* the multimodal response that makes mode averaging bite. To run on a
real dataset where egress is open, replace `make_perturbseq()` with:

    from bio_perturbations.datasets import load_norman_2019, simulation_split, apply_split
    adata = load_norman_2019(); adata, _ = simulation_split(adata); parts = apply_split(adata)
    train_reference, truth = parts["train"], parts["test"]

Everything downstream (model, XM wrapper, evaluator call) is identical.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from anndata import AnnData

from xm_core import xm_chunked_best_of_k
from bio_perturbations.evaluator import BenchmarkEvaluator
from bio_perturbations.io import make_prediction_anndata
from bio_perturbations.baselines import IdentityBaseline, MeanShiftBaseline

DEVICE = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Realistic synthetic Perturb-seq (raw counts, heterogeneous responses)
# ---------------------------------------------------------------------------
def make_perturbseq(seed, n_genes=50, n_perts=5, cells_per_cond_per_sample=150,
                    n_samples=2, program_size=8, responder_frac=0.5):
    """
    Returns a contract-compliant AnnData of raw counts with:
      obs["condition"] (perturbation label, "control" for controls),
      obs["sample_id"] (biological replicate, for pseudobulk DESeq2),
      obs["target_genes"] (knocked-down gene per perturbation),
      layers["counts"] (raw ints), var_names = gene symbols.

    Each perturbation knocks down a target gene AND drives a DE program in a
    RESPONDER subpopulation (fraction `responder_frac`) while non-responders
    shift only weakly -> a bimodal, heterogeneous response with real DEGs.
    """
    rng = np.random.default_rng(seed)
    genes = [f"g{i:02d}" for i in range(n_genes)]
    base = rng.gamma(shape=2.0, scale=6.0, size=n_genes) + 2.0  # per-gene base mean counts

    # Fixed perturbation biology (shared across seeds so train/truth agree)
    bio = np.random.default_rng(0)
    targets = [int(t) for t in bio.choice(n_genes, size=n_perts, replace=False)]
    programs, prog_lfc, weak_scale = [], [], []
    for p in range(n_perts):
        prog = bio.choice(n_genes, size=program_size, replace=False)
        lfc = bio.choice([-1.0, 1.0], size=program_size) * bio.uniform(0.8, 1.8, size=program_size)
        programs.append(prog)
        prog_lfc.append(lfc)
        weak_scale.append(0.15)  # non-responders show 15% of the responder shift

    rows, counts = [], []
    conditions = ["control"] + [f"KO_{genes[targets[p]]}" for p in range(n_perts)]
    for s in range(n_samples):
        sample_id = f"s{s+1}"
        batch = rng.normal(1.0, 0.04)  # mild per-replicate scaling
        for ci, cond in enumerate(conditions):
            n = cells_per_cond_per_sample
            for _ in range(n):
                mean = base.copy()
                tgt = ""
                if ci > 0:
                    p = ci - 1
                    tgt = genes[targets[p]]
                    mean[targets[p]] *= 0.2  # on-target knockdown
                    responder = rng.random() < responder_frac
                    scale = 1.0 if responder else weak_scale[p]
                    mean[programs[p]] *= np.exp2(prog_lfc[p] * scale)
                lam = np.clip(mean * batch, 0.05, None)
                counts.append(rng.poisson(lam))
                rows.append({"condition": cond, "sample_id": sample_id, "target_genes": tgt})

    obs = pd.DataFrame(rows, index=[f"cell_{i}" for i in range(len(rows))])
    X = np.asarray(counts, dtype=np.int64)
    adata = AnnData(X=X.astype(float), obs=obs)
    adata.var_names = genes
    adata.var["include_for_evaluation"] = True
    adata.layers["counts"] = X
    return adata


# ---------------------------------------------------------------------------
# mini-scLDM: conditional flow-matching velocity net (log1p expression space)
# ---------------------------------------------------------------------------
class CondVelocity(nn.Module):
    def __init__(self, dim, n_conditions, embed=32, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(n_conditions, embed)
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + embed, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t, t, cond):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t, self.embed(cond)], dim=-1))


def loss_calc_wrapper(model_forward, conditions, gt_samples, learning=True,
                      rand_inputs=None, rand_seeds=None):
    """Flow-matching loss (CondOTProbPath convention), compatible with XM."""
    t, cond = conditions
    with torch.set_grad_enabled(learning):
        x0, x1 = rand_inputs, gt_samples
        x_t = (1.0 - t) * x0 + t * x1
        u_t = x1 - x0
        v = model_forward(x_t, t.squeeze(-1), cond)
        per_sample = (v - u_t).pow(2).reshape(x1.shape[0], -1).mean(dim=1)
        return per_sample, None


def fit_flow(Z, cond_ids, n_conditions, best_of_k, seed, updates=3000, batch=256, lr=2e-3):
    torch.manual_seed(seed)
    model = CondVelocity(dim=Z.shape[1], n_conditions=n_conditions).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Z = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
    C = torch.tensor(cond_ids, dtype=torch.long, device=DEVICE)
    N = Z.shape[0]
    model.train()
    for _ in range(updates):
        idx = torch.randint(0, N, (batch,), device=DEVICE)
        x1, cond = Z[idx], C[idx]
        t = torch.rand(batch, 1, device=DEVICE)
        opt.zero_grad()
        losses, _ = xm_chunked_best_of_k(
            model.forward, loss_calc_wrapper, conditions=(t, cond), gt_samples=x1,
            best_of_k=best_of_k, max_chunk_bs_mult=max(best_of_k, 1),
            save_mem_mode=True, not_training=False)
        losses.mean().backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def sample_flow(model, cond_id, n, dim, n_steps=10):
    cond = torch.full((n,), cond_id, dtype=torch.long, device=DEVICE)
    x = torch.randn(n, dim, device=DEVICE)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i * dt, device=DEVICE)
        x = x + dt * model(x, t, cond)
    return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Build a prediction AnnData from a trained flow model
# ---------------------------------------------------------------------------
def flow_predictions(model, conditions, control_label, genes, mu, sd, n_gen, n_steps):
    """Generate cells per condition, invert standardisation -> non-negative
    count-scale expression, and assemble a contract-compliant prediction."""
    dim = len(genes)
    cond_to_id = {c: i for i, c in enumerate(conditions)}

    def gen(cond_label):
        z = sample_flow(model, cond_to_id[cond_label], n_gen, dim, n_steps=n_steps)
        expr = np.expm1(z * sd + mu)          # invert standardised log1p
        return np.clip(expr, 0.0, None)

    matrices = {c: gen(c) for c in conditions if c != control_label}
    control = gen(control_label)
    return make_prediction_anndata(matrices, genes, control=control)


def baseline_predictions(model_obj, train_ref, perts, control_label, genes, n_gen):
    """Predict with a bio-perturbations baseline and wrap as prediction AnnData."""
    fitted = model_obj.fit(train_ref)
    pred = fitted.predict(list(perts))
    # baselines emit a contract AnnData; clip to non-negative (count expression
    # can't be negative, and mean_shift can push knocked-down genes below zero).
    # The evaluator refuses to clip silently, so we do it explicitly here.
    pred.X = np.clip(np.asarray(pred.X, dtype=float), 0.0, None)
    return pred


# ---------------------------------------------------------------------------
def summarise(report):
    """Mean over perturbations of the headline biological metrics."""
    rows = report["per_perturbation"]
    def col(getter):
        vals = [getter(r) for r in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "pcc_delta": col(lambda r: r.aggregate.get("pcc_delta")),
        "mse_delta": col(lambda r: r.aggregate.get("mse_delta")),
        "deg_dir_recall@20": col(lambda r: r.deg.get("directional_recall_at_20")),
        "deg_dir_recall@50": col(lambda r: r.deg.get("directional_recall_at_50")),
        "energy_distance": col(lambda r: r.distribution.get("energy_distance")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_gen", type=int, default=300)
    args = ap.parse_args()

    # Fixed benchmark data: train_reference + held-out truth (same biology)
    train_ref = make_perturbseq(seed=0)
    truth = make_perturbseq(seed=1)
    genes = list(train_ref.var_names)
    conditions = list(pd.unique(train_ref.obs["condition"]))       # ["control", "KO_..."]
    perts = [c for c in conditions if c != "control"]

    # Standardise log1p(counts) for the flow model
    Xlog = np.log1p(np.asarray(train_ref.X, dtype=float))
    mu, sd = Xlog.mean(0), Xlog.std(0) + 1e-6
    Z = (Xlog - mu) / sd
    cond_ids = np.array([conditions.index(c) for c in train_ref.obs["condition"]])

    evaluator = BenchmarkEvaluator.from_anndata(train_ref, n_pca_components=20,
                                                min_matched_samples=2)

    results = {}
    # --- bio-perturbations reference baselines ---
    for name, obj in [("identity", IdentityBaseline()), ("mean_shift", MeanShiftBaseline())]:
        pred = baseline_predictions(obj, train_ref, perts, "control", genes, args.n_gen)
        rep = evaluator.evaluate_anndata(pred, truth)
        results[name] = summarise(rep)
        print(f"[baseline {name}] " + " ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    # --- flow models: baseline K=1 vs XM K>1, averaged over seeds ---
    for k in args.ks:
        per_seed = []
        for s in args.seeds:
            model = fit_flow(Z, cond_ids, len(conditions), best_of_k=k, seed=s, updates=args.updates)
            pred = flow_predictions(model, conditions, "control", genes, mu, sd,
                                    n_gen=args.n_gen, n_steps=args.n_steps)
            rep = evaluator.evaluate_anndata(pred, truth)
            per_seed.append(summarise(rep))
            tag = "baseline flow (K=1)" if k == 1 else f"XM flow (K={k})"
            print(f"[{tag} seed={s}] " + " ".join(f"{kk}={vv:.4f}" for kk, vv in per_seed[-1].items()))
        agg = {kk: float(np.mean([ps[kk] for ps in per_seed])) for kk in per_seed[0]}
        agg_std = {kk + "_std": float(np.std([ps[kk] for ps in per_seed])) for kk in per_seed[0]}
        label = "flow_K1" if k == 1 else f"flow_XM_K{k}"
        results[label] = {**agg, **agg_std}
        print(f"==> {label} AGG: " + " ".join(f"{kk}={agg[kk]:.4f}" for kk in agg) + "\n")

    out = {"config": vars(args), "conditions": conditions, "results": results,
           "note": "synthetic Perturb-seq (real dataset egress blocked in sandbox); "
                   "metrics from bio_perturbations.evaluator (real)"}
    with open(os.path.join(HERE, "scldm_bio_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote scldm_bio_results.json")


if __name__ == "__main__":
    main()
