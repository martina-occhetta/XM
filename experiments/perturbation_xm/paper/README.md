# AI4DD @ NeurIPS 2026 — paper draft

5-page skeleton for the perturbation-response / Explorative-Modeling submission.
The scientific content, tables, and figure are wired to the experiment outputs in
`../` (JSON + PNG). `\TODO{...}` marks every gap — almost all are "swap synthetic
numbers for real-data numbers."

## Format (from the CfP)

- **Venue:** AI4DD — AI for Drug Discovery: Bridging the Translation Gap, NeurIPS 2026 (Sydney).
- **Deadline:** 29 Aug 2026.
- **Length:** up to **5 pages** (full paper) or **2 pages** (tiny paper), refs/appendix excluded.
- **Style:** NeurIPS 2026 style, **anonymized**: `\usepackage{neurips_2026}` (no options).
  Drop `neurips_2026.sty` + `neurips_2026.bst` next to `main.tex` (from the workshop/NeurIPS site).

## Build

```bash
cd paper
mkdir -p figs
cp ../scldm_ablation_synthetic_vae.png figs/ablation.png     # main ablation figure
# (optional) cp ../perturbation_xm_samples.png figs/toy.png   # the 2-D mechanism figure
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Fill-in checklist (do the real-data run first)

Run on ≥1 real dataset in an egress-open env, then transcribe numbers:

```bash
pip install -e /path/to/bio-perturbations".[prep,datasets]"
# main table numbers (per-model biological metrics, incl. DEG + pathway):
python ../scldm_bio_eval.py --dataset replogle_2022_k562 --space nbvae \
    --n-hvg 2000 --max-perts 20 --seeds 0 1 2 --ks 1 4 --strict-dataset \
    --gene-sets your_programs.json --out ../results_real.json
# ablation figure + paired CIs:
python ../scldm_ablation.py --dataset replogle_2022_k562 --space nbvae \
    --ks 1 2 4 8 --steps 2 4 10 --seeds 0 1 2 3 4 --strict-dataset --plot
```

Then:

- [ ] `tab:main` — replace synthetic numbers with real (all rows incl. **scGen**).
- [ ] Biological-signal paragraph — confirm DEG recall/Jaccard/effect-$r$ and
      pathway $\rho$ direction on real data (the central claim).
- [ ] `fig:ablation` — regenerate `figs/ablation.png` from the real ablation run.
- [ ] Precision–calibration paragraph — state real-data direction/magnitude of the MSE-Δ cost.
- [ ] Latent-space robustness — move the 3-space table (logexpr/vae/nbvae) into an appendix.
- [ ] Baselines — add an established model (GEARS/CPA) if time permits.
- [ ] Pathway gene sets — supply `--gene-sets` (e.g. MSigDB Hallmark / the DEGs of
      each perturbation); on synthetic the injected programs are used automatically.
- [ ] Verify every `references.bib` entry (several are placeholders).
- [ ] Fill author/affiliation only in the camera-ready; keep anonymous for submission.

## Tiny-paper (2pg) fallback

If real-data results are thin by ~24 Aug: cut to Intro + Method (1 short §) +
one results figure (the ablation) + the DEG/pathway table + a limitations
sentence. Keep the honest precision–calibration framing — it is the AI4DD angle.

## Source of every number

| Paper element | Produced by |
| --- | --- |
| `tab:main` (E-dist, PCC/MSE-Δ, DEG, pathway) | `../scldm_bio_eval.py … --gene-sets` → `*_results_*.json` |
| `fig:ablation` (K × steps, CIs) | `../scldm_ablation.py --plot` → `scldm_ablation_*.png/.json` |
| 3-latent-space robustness | `../scldm_{vae,nbvae}_run.log`, README §8a |
| Mechanism figure (2-D) | `../perturbation_xm_samples.png` |
