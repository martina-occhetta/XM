"""
End-to-end integration: Explorative Modeling (best-of-K exploration) applied to
an scLDM-shaped conditional generative model, scored on *biological* signal with
the `bio-perturbations` evaluator (DEG recovery, E-distance, PCC-Δ), against that
framework's own Identity / MeanShift baselines.

This harness has two axes, both selectable from the CLI:

  --dataset  synthetic | norman_2019 | replogle_2022_k562 | adamson_2016 | ...
             `synthetic` = a self-contained Perturb-seq generator (default, always
             runnable). Any other id is loaded through `bio_perturbations.datasets`
             (pertpy) -- real Perturb-seq. See "Running on real data" below.

  --space    logexpr | vae | nbvae
             `logexpr` = flow runs in standardised log1p expression space (default).
             `vae`     = a small scLDM-style Gaussian VAE (on log1p) is trained first
             and the flow runs in its LATENT space (encode -> flow+XM in latent ->
             decode). A conditional latent generative model over a learned code.
             `nbvae`   = the same, but with scVI's negative-binomial count decoder
             (library-scaled softmax proportions + per-gene dispersion) -- the
             count-likelihood generative model scLDM/scVI actually use.
             The XM wrapper and evaluation are unchanged across all three.

The exploration engine is the repo's real `xm_chunked_best_of_k` (vendored verbatim
in xm_core.py). Baseline = `--ks 1`; XM = `--ks 4` (etc). Everything is CPU-friendly.

Running on real data
--------------------
Real Perturb-seq loaders live in `bio_perturbations.datasets` (pertpy, gated behind
the `[datasets]` extra). Their download hosts (figshare, cellxgene) are commonly
egress-blocked; when a load fails this script prints why and (unless --strict-dataset)
falls back to the synthetic generator with a loud banner, so a first run never hard-
crashes. In an egress-open environment:

    pip install -e /path/to/bio-perturbations".[prep,datasets]"
    python scldm_bio_eval.py --dataset norman_2019 --space vae \\
        --n-hvg 2000 --max-perts 20 --max-cells-per-cond 400 --ks 1 4

The task is *distribution recovery on seen perturbations* (does the model reproduce
the held-out perturbed cell population, heterogeneity included) -- the mode-averaging
question. We therefore hold out CELLS within each perturbation (not whole
perturbations); the model conditions on a learned per-perturbation embedding.
Unseen-perturbation generalisation is a different task needing perturbation features
(see bio_perturbations `LinearBaseline`) and is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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


# ===========================================================================
# Data
# ===========================================================================
def make_perturbseq(seed, n_genes=50, n_perts=5, cells_per_cond_per_sample=150,
                    n_samples=2, program_size=8, responder_frac=0.5):
    """Realistic synthetic Perturb-seq (raw counts, heterogeneous responses).

    Each perturbation knocks down a target gene AND drives a DE program in a
    RESPONDER subpopulation while non-responders shift only weakly -> a bimodal,
    heterogeneous response with genuine downstream DEGs. Contract-compliant:
    obs[condition/sample_id/target_genes], layers["counts"], var_names.
    """
    rng = np.random.default_rng(seed)
    genes = [f"g{i:02d}" for i in range(n_genes)]
    base = rng.gamma(shape=2.0, scale=6.0, size=n_genes) + 2.0

    bio = np.random.default_rng(0)  # fixed biology so train/truth agree across seeds
    targets = [int(t) for t in bio.choice(n_genes, size=n_perts, replace=False)]
    programs, prog_lfc, weak_scale = [], [], []
    for p in range(n_perts):
        programs.append(bio.choice(n_genes, size=program_size, replace=False))
        prog_lfc.append(bio.choice([-1.0, 1.0], size=program_size) * bio.uniform(0.8, 1.8, size=program_size))
        weak_scale.append(0.15)

    rows, counts = [], []
    conditions = ["control"] + [f"KO_{genes[targets[p]]}" for p in range(n_perts)]
    for s in range(n_samples):
        sample_id = f"s{s+1}"
        batch = rng.normal(1.0, 0.04)
        for ci, cond in enumerate(conditions):
            for _ in range(cells_per_cond_per_sample):
                mean = base.copy()
                tgt = ""
                if ci > 0:
                    p = ci - 1
                    tgt = genes[targets[p]]
                    mean[targets[p]] *= 0.2
                    scale = 1.0 if rng.random() < responder_frac else weak_scale[p]
                    mean[programs[p]] *= np.exp2(prog_lfc[p] * scale)
                counts.append(rng.poisson(np.clip(mean * batch, 0.05, None)))
                rows.append({"condition": cond, "sample_id": sample_id, "target_genes": tgt})

    obs = pd.DataFrame(rows, index=[f"cell_{i}" for i in range(len(rows))])
    X = np.asarray(counts, dtype=np.int64)
    adata = AnnData(X=X.astype(float), obs=obs)
    adata.var_names = genes
    adata.var["include_for_evaluation"] = True
    adata.layers["counts"] = X
    return adata


def perturbseq_gene_sets(n_genes=50, n_perts=5, program_size=8):
    """Reconstruct the DE programs injected by make_perturbseq as ground-truth
    gene sets {program_name: [gene, ...]}, replaying the same fixed `bio` RNG.

    Passed to the evaluator as `gene_sets` so the pathway metrics measure whether
    the model recovers each perturbation's known transcriptional program -- a
    self-contained pathway-recovery test that needs no external enrichment DB.
    """
    genes = [f"g{i:02d}" for i in range(n_genes)]
    bio = np.random.default_rng(0)
    targets = [int(t) for t in bio.choice(n_genes, size=n_perts, replace=False)]
    gene_sets = {}
    for p in range(n_perts):
        prog = bio.choice(n_genes, size=program_size, replace=False)
        _ = bio.choice([-1.0, 1.0], size=program_size) * bio.uniform(0.8, 1.8, size=program_size)
        gene_sets[f"program_KO_{genes[targets[p]]}"] = [genes[i] for i in prog]
    return gene_sets


def _dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)


def _assign_pseudoreplicates(adata, n_reps, seed):
    """Ensure >=n_reps biological-replicate labels per condition for DESeq2.

    Real Perturb-seq often lacks replicates; DESeq2 pseudobulk needs >=2 samples
    per condition. We partition each condition's cells into `n_reps` random groups.
    NOTE: pseudo-replicates underestimate true biological variance -- fine for a
    demo / relative model comparison, not for absolute significance claims.
    """
    rng = np.random.default_rng(seed)
    sample_id = np.empty(adata.n_obs, dtype=object)
    cond = adata.obs["condition"].to_numpy()
    for c in pd.unique(cond):
        idx = np.flatnonzero(cond == c)
        rng.shuffle(idx)
        for j, cell in enumerate(idx):
            sample_id[cell] = f"rep{j % n_reps + 1}"
    adata.obs["sample_id"] = sample_id
    return adata


def cell_level_holdout(adata, test_fraction, seed):
    """Split CELLS within each condition into (train_reference, truth).

    Keeps the full perturbation vocabulary in both splits -- the distribution-
    recovery setup the flow model (learned per-pert embedding) is built for.
    """
    rng = np.random.default_rng(seed)
    cond = adata.obs["condition"].to_numpy()
    train_mask = np.zeros(adata.n_obs, dtype=bool)
    for c in pd.unique(cond):
        idx = np.flatnonzero(cond == c)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        train_mask[idx[n_test:]] = True
    train = adata[train_mask].copy()
    truth = adata[~train_mask].copy()
    return train, truth


def _subset_for_tractability(adata, n_hvg, max_perts, max_cells_per_cond, seed):
    """Optionally shrink a large real dataset so it trains on CPU in minutes.

    - keep control + the `max_perts` perturbations with the most cells
    - keep the top `n_hvg` high-variance genes UNION the selected perts' targets
    - subsample each condition to `max_cells_per_cond` cells
    """
    rng = np.random.default_rng(seed)
    cond = adata.obs["condition"].astype(str)

    # perturbation selection
    counts_per = cond[cond != "control"].value_counts()
    keep_perts = list(counts_per.index[:max_perts]) if max_perts else list(counts_per.index)
    keep_conds = ["control"] + keep_perts
    adata = adata[cond.isin(keep_conds)].copy()

    # gene selection (HVG on log1p) unioned with on-target genes
    if n_hvg and n_hvg < adata.n_vars:
        Y = np.log1p(_dense(adata.layers.get("counts", adata.X)))
        hvg = np.argsort(Y.var(axis=0))[::-1][:n_hvg]
        targets = set()
        for t in adata.obs["target_genes"].astype(str):
            targets.update(x for x in t.replace("+", " ").split() if x)
        target_idx = [i for i, g in enumerate(adata.var_names) if g in targets]
        keep_genes = np.union1d(hvg, np.asarray(target_idx, dtype=int)) if target_idx else hvg
        adata = adata[:, np.sort(keep_genes)].copy()

    # per-condition cell cap
    if max_cells_per_cond:
        cond = adata.obs["condition"].to_numpy()
        keep = []
        for c in pd.unique(cond):
            idx = np.flatnonzero(cond == c)
            if len(idx) > max_cells_per_cond:
                idx = rng.choice(idx, size=max_cells_per_cond, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        adata = adata[keep].copy()
    return adata


def load_real_dataset(name, *, n_hvg, max_perts, max_cells_per_cond,
                      n_pseudoreplicates, test_fraction, cache_dir, seed):
    """Load a real pertpy dataset via bio_perturbations and prepare 4-state eval.

    Returns (train_reference, truth), both contract-ready with pseudo-replicate
    sample_id for DESeq2. Raises on any load failure (caller decides fallback).
    """
    from bio_perturbations.datasets import load_dataset
    adata = load_dataset(name, cache_dir=cache_dir, require_raw_counts=True)
    # ensure a raw-count layer for DESeq2 (loader sets it, but be defensive)
    if "counts" not in adata.layers:
        adata.layers["counts"] = _dense(adata.X)
    adata = _subset_for_tractability(adata, n_hvg, max_perts, max_cells_per_cond, seed)
    adata.var["include_for_evaluation"] = True
    train, truth = cell_level_holdout(adata, test_fraction=test_fraction, seed=seed)
    _assign_pseudoreplicates(train, n_pseudoreplicates, seed=seed)
    _assign_pseudoreplicates(truth, n_pseudoreplicates, seed=seed + 1)
    return train, truth


def get_data(args):
    """Dispatch to synthetic or real data; returns (train_ref, truth, source_note)."""
    if args.dataset == "synthetic":
        return make_perturbseq(seed=0), make_perturbseq(seed=1), "synthetic Perturb-seq"
    try:
        train, truth = load_real_dataset(
            args.dataset, n_hvg=args.n_hvg, max_perts=args.max_perts,
            max_cells_per_cond=args.max_cells_per_cond,
            n_pseudoreplicates=args.n_pseudoreplicates,
            test_fraction=args.test_fraction, cache_dir=args.cache_dir, seed=0)
        return train, truth, f"real dataset '{args.dataset}' (bio_perturbations/pertpy)"
    except Exception as e:  # noqa: BLE001 - want any failure (network/import/etc)
        msg = f"{type(e).__name__}: {e}"
        if args.strict_dataset:
            print(f"ERROR: failed to load real dataset '{args.dataset}': {msg}", file=sys.stderr)
            raise
        print("\n" + "!" * 78)
        print(f"WARNING: could not load real dataset '{args.dataset}':\n  {msg}")
        print("Falling back to SYNTHETIC Perturb-seq. To run on real data, use an")
        print("egress-open environment with `[datasets]` installed, or pass --strict-dataset.")
        print("!" * 78 + "\n")
        return make_perturbseq(seed=0), make_perturbseq(seed=1), \
            f"synthetic (fallback; '{args.dataset}' unavailable: {type(e).__name__})"


# ===========================================================================
# Representation spaces: logexpr (identity-ish) and VAE latent (scLDM-style)
# ===========================================================================
class LogExprSpace:
    """Standardised log1p expression. Flow runs directly in gene space."""

    name = "logexpr"

    def fit(self, train_counts):
        Y = np.log1p(np.asarray(train_counts, dtype=float))
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-6
        self.dim = Y.shape[1]
        return self

    def encode(self, counts):
        return (np.log1p(np.asarray(counts, dtype=float)) - self.mu) / self.sd

    def decode(self, Z):
        return np.clip(np.expm1(Z * self.sd + self.mu), 0.0, None)


class VAE(nn.Module):
    """Small Gaussian VAE over standardised log1p expression (scLDM stand-in)."""

    def __init__(self, n_genes, latent_dim, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, n_genes))

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


class VAESpace:
    """Train a VAE on train cells; flow then runs in the VAE latent (encoder mean)."""

    name = "vae"

    def __init__(self, latent_dim=16, epochs=60, beta=1e-3, lr=1e-3, batch=256, seed=0):
        self.latent_dim, self.epochs, self.beta = latent_dim, epochs, beta
        self.lr, self.batch, self.seed = lr, batch, seed

    def fit(self, train_counts):
        torch.manual_seed(self.seed)
        Y = np.log1p(np.asarray(train_counts, dtype=float))
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-6
        Ystd = torch.tensor((Y - self.mu) / self.sd, dtype=torch.float32, device=DEVICE)
        n_genes = Ystd.shape[1]
        self.vae = VAE(n_genes, self.latent_dim).to(DEVICE)
        opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        N = Ystd.shape[0]
        self.vae.train()
        steps = max(1, N // self.batch)
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=DEVICE)
            for b in range(steps):
                xb = Ystd[perm[b * self.batch:(b + 1) * self.batch]]
                xhat, mu, logvar = self.vae(xb)
                recon = (xhat - xb).pow(2).mean()
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                loss = recon + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
        self.vae.eval()
        self.dim = self.latent_dim
        return self

    @torch.no_grad()
    def encode(self, counts):
        Y = (np.log1p(np.asarray(counts, dtype=float)) - self.mu) / self.sd
        mu, _ = self.vae.encode(torch.tensor(Y, dtype=torch.float32, device=DEVICE))
        return mu.cpu().numpy()

    @torch.no_grad()
    def decode(self, Z):
        Yhat = self.vae.decode(torch.tensor(Z, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        return np.clip(np.expm1(Yhat * self.sd + self.mu), 0.0, None)


class NBVAE(nn.Module):
    """scVI-style negative-binomial VAE.

    Generative model (per cell): z ~ N(0,I); the decoder maps z to gene
    proportions rho = softmax(dec(z)); with observed library size l the mean is
    mu = l * rho, and counts x ~ NB(mu, theta) with a per-gene inverse-dispersion
    theta. This is the count-likelihood decoder scLDM/scVI actually use, versus
    the Gaussian-on-log1p VAE above.
    """

    def __init__(self, n_genes, latent_dim, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, n_genes))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))  # per-gene inverse dispersion

    def encode(self, x_enc):
        h = self.enc(x_enc)
        return self.fc_mu(h), self.fc_logvar(h)

    def rho(self, z):
        return torch.softmax(self.dec(z), dim=-1)  # gene proportions, sum to 1


def nb_neg_log_likelihood(x, mu, theta, eps=1e-8):
    """-log NB(x; mean=mu, inverse-dispersion=theta), summed over genes."""
    theta = theta + eps
    mu = mu + eps
    log_theta_mu = torch.log(theta + mu)
    ll = (theta * (torch.log(theta) - log_theta_mu)
          + x * (torch.log(mu) - log_theta_mu)
          + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0))
    return -ll.sum(dim=-1)


class NBVAESpace:
    """scVI-style NB-VAE; flow runs in the latent, decode returns the NB mean.

    Drop-in alternative to `VAESpace` (`--space nbvae`). Encoder input is
    standardised log1p counts (for stable optimisation); the decoder/likelihood
    operate on raw counts with an observed library size, exactly as scVI does.
    """

    name = "nbvae"

    def __init__(self, latent_dim=16, epochs=80, beta=1e-3, lr=1e-3, batch=256, seed=0):
        self.latent_dim, self.epochs, self.beta = latent_dim, epochs, beta
        self.lr, self.batch, self.seed = lr, batch, seed

    def _enc_input(self, counts):
        Y = np.log1p(np.asarray(counts, dtype=float))
        return (Y - self.mu) / self.sd

    def fit(self, train_counts):
        torch.manual_seed(self.seed)
        X = np.asarray(train_counts, dtype=float)
        Y = np.log1p(X)
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-6
        self.lib_scale = float(np.median(X.sum(1)))  # library size for generation
        Xenc = torch.tensor((Y - self.mu) / self.sd, dtype=torch.float32, device=DEVICE)
        Xcount = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        lib = Xcount.sum(dim=1, keepdim=True)  # observed per-cell library
        n_genes = Xenc.shape[1]
        self.vae = NBVAE(n_genes, self.latent_dim).to(DEVICE)
        opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        N = Xenc.shape[0]
        steps = max(1, N // self.batch)
        self.vae.train()
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=DEVICE)
            for b in range(steps):
                sel = perm[b * self.batch:(b + 1) * self.batch]
                xe, xc, lb = Xenc[sel], Xcount[sel], lib[sel]
                mu_z, logvar = self.vae.encode(xe)
                z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar)
                nb_mean = lb * self.vae.rho(z)
                recon = nb_neg_log_likelihood(xc, nb_mean, torch.exp(self.vae.log_theta)).mean()
                kl = -0.5 * (1 + logvar - mu_z.pow(2) - logvar.exp()).sum(1).mean()
                loss = recon + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
        self.vae.eval()
        self.dim = self.latent_dim
        return self

    @torch.no_grad()
    def encode(self, counts):
        xe = torch.tensor(self._enc_input(counts), dtype=torch.float32, device=DEVICE)
        return self.vae.encode(xe)[0].cpu().numpy()

    @torch.no_grad()
    def decode(self, Z):
        rho = self.vae.rho(torch.tensor(Z, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        return self.lib_scale * rho  # NB mean at the median library size (non-negative)


def build_space(name, latent_dim, vae_epochs, seed):
    if name == "logexpr":
        return LogExprSpace()
    if name == "nbvae":
        return NBVAESpace(latent_dim=latent_dim, epochs=vae_epochs, seed=seed)
    if name == "vae":
        return VAESpace(latent_dim=latent_dim, epochs=vae_epochs, seed=seed)
    raise ValueError(f"unknown space {name!r}")


# ===========================================================================
# Conditional flow (mini-scLDM) + XM exploration
# ===========================================================================
class CondVelocity(nn.Module):
    def __init__(self, dim, n_conditions, embed=32, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(n_conditions, embed)
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + embed, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim))

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


# ===========================================================================
# Predict + evaluate
# ===========================================================================
def flow_predictions(model, space, conditions, control_label, genes, n_gen, n_steps):
    cond_to_id = {c: i for i, c in enumerate(conditions)}

    def gen(c):
        Z = sample_flow(model, cond_to_id[c], n_gen, space.dim, n_steps=n_steps)
        return space.decode(Z)  # -> non-negative count-scale expression

    matrices = {c: gen(c) for c in conditions if c != control_label}
    control = gen(control_label)
    return make_prediction_anndata(matrices, genes, control=control)


def baseline_predictions(model_obj, train_ref, perts):
    pred = model_obj.fit(train_ref).predict(list(perts))
    # count expression can't be negative; evaluator refuses to clip silently.
    pred.X = np.clip(np.asarray(pred.X, dtype=float), 0.0, None)
    return pred


def latent_shift_predictions(space, train_ref, conditions, control_label, genes, n_gen, seed=0):
    """scGen-style latent vector arithmetic (Lotfollahi et al. 2019).

    Predict a perturbed cell as decode(encode(control_cell) + delta_p), where
    delta_p is the mean latent shift of perturbation p over controls. A strong,
    non-generative reference: it captures the mean response and inherits control
    heterogeneity, but cannot invent NEW modes -- so contrasting it with the
    flow+XM model isolates what the generative model adds. In a VAE latent this
    is scGen proper; in `logexpr` space it reduces to a log-space mean shift.
    """
    rng = np.random.default_rng(seed)
    counts = _dense(train_ref.layers.get("counts", train_ref.X))
    cond = train_ref.obs["condition"].to_numpy()
    Z = space.encode(counts)
    ctrl_idx = np.flatnonzero(cond == control_label)
    ctrl_mean = Z[ctrl_idx].mean(0)

    def pred_for(c):
        delta = Z[np.flatnonzero(cond == c)].mean(0) - ctrl_mean
        sel = rng.choice(ctrl_idx, size=n_gen, replace=True)
        return space.decode(Z[sel] + delta)

    matrices = {c: pred_for(c) for c in conditions if c != control_label}
    control = space.decode(Z[rng.choice(ctrl_idx, size=n_gen, replace=True)])
    return make_prediction_anndata(matrices, genes, control=control)


def per_perturbation_energy(report):
    """{perturbation: E-distance} from an evaluator report (for paired CIs)."""
    return {r.perturbation: r.distribution.get("energy_distance")
            for r in report["per_perturbation"]
            if r.distribution.get("energy_distance") is not None}


def summarise(report):
    """Mean-over-perturbations of the headline metrics across all four families:
    aggregate (mean shift), DEG recovery, pathway/program recovery, distribution."""
    rows = report["per_perturbation"]

    def col(getter):
        vals = []
        for r in rows:
            try:
                v = getter(r)
            except Exception:
                v = None
            if v is not None and np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    out = {
        # aggregate (mean-shift fidelity)
        "pcc_delta": col(lambda r: r.aggregate.get("pcc_delta")),
        "mse_delta": col(lambda r: r.aggregate.get("mse_delta")),
        "r2_delta": col(lambda r: r.aggregate.get("r2_delta")),
        # DEG recovery (downstream biology)
        "deg_dir_recall@20": col(lambda r: r.deg.get("directional_recall_at_20")),
        "deg_dir_recall@50": col(lambda r: r.deg.get("directional_recall_at_50")),
        "deg_jaccard@20": col(lambda r: r.deg.get("jaccard_at_20")),
        "deg_effect_pearson": col(lambda r: r.deg.get("effect_pearson_true_degs")),
        # distribution (heterogeneity)
        "energy_distance": col(lambda r: r.distribution.get("energy_distance")),
    }
    # pathway / program recovery (only present when gene_sets were supplied)
    if rows and rows[0].pathway:
        out["pathway_jaccard_up"] = col(lambda r: r.pathway.get("jaccard_up"))
        out["pathway_jaccard_down"] = col(lambda r: r.pathway.get("jaccard_down"))
        out["pathway_spearman_up"] = col(lambda r: r.pathway.get("spearman_up"))
        out["pathway_spearman_down"] = col(lambda r: r.pathway.get("spearman_down"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic",
                    help="'synthetic' or a bio_perturbations dataset id (norman_2019, ...)")
    ap.add_argument("--space", default="logexpr", choices=["logexpr", "vae", "nbvae"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_gen", type=int, default=300)
    # VAE
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--vae-epochs", type=int, default=60)
    # real-data prep
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--max-perts", type=int, default=20)
    ap.add_argument("--max-cells-per-cond", type=int, default=400)
    ap.add_argument("--n-pseudoreplicates", type=int, default=2)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--strict-dataset", action="store_true",
                    help="fail (don't fall back to synthetic) if the real load fails")
    ap.add_argument("--out", default=None, help="results json path")
    ap.add_argument("--gene-sets", default=None,
                    help="JSON {name: [gene,...]} for pathway metrics (real data). "
                         "For --dataset synthetic the injected DE programs are used automatically.")
    args = ap.parse_args()

    train_ref, truth, source_note = get_data(args)
    genes = list(train_ref.var_names)
    conditions = list(pd.unique(train_ref.obs["condition"]))
    perts = [c for c in conditions if c != "control"]
    print(f"[data] {source_note}: {train_ref.n_obs} train / {truth.n_obs} truth cells, "
          f"{len(genes)} genes, {len(perts)} perturbations")

    # representation space (logexpr or trained VAE latent)
    train_counts = _dense(train_ref.layers.get("counts", train_ref.X))
    space = build_space(args.space, args.latent_dim, args.vae_epochs, seed=0).fit(train_counts)
    print(f"[space] {space.name}: flow dim = {space.dim}")
    Z = space.encode(train_counts)
    cond_ids = np.array([conditions.index(c) for c in train_ref.obs["condition"]])

    # gene sets for pathway/program recovery: injected programs (synthetic) or a
    # user JSON (real data); None -> pathway metrics are skipped.
    gene_sets = None
    if args.gene_sets:
        with open(args.gene_sets) as f:
            gene_sets = json.load(f)
    elif source_note.startswith("synthetic"):  # real data can fall back to synthetic
        gene_sets = perturbseq_gene_sets()
    if gene_sets:
        print(f"[pathway] {len(gene_sets)} gene sets -> pathway recovery metrics on")

    n_pca = int(min(20, len(genes) - 1, train_ref.n_obs - 1))
    evaluator = BenchmarkEvaluator.from_anndata(train_ref, n_pca_components=n_pca,
                                                min_matched_samples=2, gene_sets=gene_sets)

    results = {}
    for name, obj in [("identity", IdentityBaseline()), ("mean_shift", MeanShiftBaseline())]:
        rep = evaluator.evaluate_anndata(baseline_predictions(obj, train_ref, perts), truth)
        results[name] = summarise(rep)
        print(f"[baseline {name}] " + " ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    for k in args.ks:
        per_seed = []
        for s in args.seeds:
            model = fit_flow(Z, cond_ids, len(conditions), best_of_k=k, seed=s, updates=args.updates)
            pred = flow_predictions(model, space, conditions, "control", genes,
                                    n_gen=args.n_gen, n_steps=args.n_steps)
            per_seed.append(summarise(evaluator.evaluate_anndata(pred, truth)))
            tag = "baseline flow (K=1)" if k == 1 else f"XM flow (K={k})"
            print(f"[{tag} seed={s}] " + " ".join(f"{kk}={vv:.4f}" for kk, vv in per_seed[-1].items()))
        agg = {kk: float(np.mean([ps[kk] for ps in per_seed])) for kk in per_seed[0]}
        agg.update({kk + "_std": float(np.std([ps[kk] for ps in per_seed])) for kk in per_seed[0]})
        label = "flow_K1" if k == 1 else f"flow_XM_K{k}"
        results[label] = agg
        print(f"==> {label} AGG: " + " ".join(f"{kk}={agg[kk]:.4f}" for kk in list(agg)[:5]) + "\n")

    out = {"config": vars(args), "source": source_note, "space": space.name,
           "flow_dim": int(space.dim), "conditions": conditions, "results": results}
    out_path = args.out or os.path.join(HERE, f"scldm_bio_results_{args.dataset}_{space.name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
