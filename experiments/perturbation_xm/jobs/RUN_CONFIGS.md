# Run configs: Adamson, Norman, Replogle × Forward/Reverse XM

Commands for the real-data runs. 

All jobs read the cache written by the download step, so download
each dataset **once**. Edit `-A`/`-p` for your cluster (here: `pilot`/`compute`
for CPU; `pilot_andrena`/`andrena` + `--gres=gpu:1` for GPU).

Set the space per your framing:
- `SPACE=nbvae`  — the scLDM (latent diffusion) setting; where XM is expected to matter.
- `SPACE=logexpr` — non-latent control (flow in gene space); best absolute E-distance.

`XM_DIRECTION=forward` (default) explores noises; `XM_DIRECTION=reverse` explores
data targets (keeps the noise marginal Gaussian; better calibration in early tests).

## 1. Download (once per dataset)

Memory scales with dataset size (`load_dataset` loads + copies the count matrix).

```bash
# Adamson (~24k cells) — small; 32G is ample
DATASETS="adamson_2016" sbatch -A pilot -p compute -n 4 --mem-per-cpu=8G \
  experiments/perturbation_xm/jobs/01_download_data.sh

# Norman (~111k cells) — already cached if you ran it; ~64G
DATASETS="norman_2019" sbatch -A pilot -p compute -n 4 --mem-per-cpu=16G \
  experiments/perturbation_xm/jobs/01_download_data.sh

# Replogle K562 essential (~300k cells, genome-scale) — BIG: use highmem, ~192G
DATASETS="replogle_2022_k562" sbatch -A pilot -p highmem -n 8 --mem-per-cpu=24G \
  experiments/perturbation_xm/jobs/01_download_data.sh
```
If a compute node has no internet, run the download on a login node instead (see
`jobs/README.md`), then the compute jobs below read the cache.

## 2. Bio-eval (per-model biological metrics: E-distance, DEG, pathway, PCC/MSE-Δ)

```bash
# Adamson — Forward vs Reverse XM, scLDM (nbvae) setting
DATASET=adamson_2016 SPACE=nbvae LATENT_DIM=64 VAE_EPOCHS=200 XM_DIRECTION=forward \
  sbatch -A pilot -p compute -n 8 --mem-per-cpu=8G experiments/perturbation_xm/jobs/02_bio_eval.sh
DATASET=adamson_2016 SPACE=nbvae LATENT_DIM=64 VAE_EPOCHS=200 XM_DIRECTION=reverse \
  sbatch -A pilot -p compute -n 8 --mem-per-cpu=8G experiments/perturbation_xm/jobs/02_bio_eval.sh

# Replogle — Forward vs Reverse (raise MAX_PERTS: many perturbations available)
DATASET=replogle_2022_k562 SPACE=nbvae LATENT_DIM=64 VAE_EPOCHS=200 MAX_PERTS=40 \
  XM_DIRECTION=forward sbatch -A pilot -p compute -n 8 --mem-per-cpu=12G \
  experiments/perturbation_xm/jobs/02_bio_eval.sh
DATASET=replogle_2022_k562 SPACE=nbvae LATENT_DIM=64 VAE_EPOCHS=200 MAX_PERTS=40 \
  XM_DIRECTION=reverse sbatch -A pilot -p compute -n 8 --mem-per-cpu=12G \
  experiments/perturbation_xm/jobs/02_bio_eval.sh
```
Outputs: `results/bioeval_<dataset>_<space>[_reverse].json`.

## 3. Ablation (K×steps grid + bootstrap CIs + figure)

```bash
# Adamson — Forward and Reverse XM ablations
DATASET=adamson_2016 SPACE=nbvae LATENT_DIM=64 VAE_EPOCHS=200 XM_DIRECTION=forward \
  sbatch -A pilot -p compute -n 16 --mem-per-cpu=8G experiments/perturbation_xm/jobs/03_ablation.sh
DATASET=adamson_2016 SPACE=nbvae LATENT_DIM=64 VAE_EPOCHS=200 XM_DIRECTION=reverse \
  sbatch -A pilot -p compute -n 16 --mem-per-cpu=8G experiments/perturbation_xm/jobs/03_ablation.sh

# Replogle — same, larger panel
DATASET=replogle_2022_k562 SPACE=nbvae LATENT_DIM=64 VAE_EPOCHS=200 MAX_PERTS=40 \
  XM_DIRECTION=forward sbatch -A pilot -p compute -n 16 --mem-per-cpu=12G \
  experiments/perturbation_xm/jobs/03_ablation.sh
```
Outputs: `results/scldm_ablation_<dataset>_<space>[_reverse].{json,png}` (png also copied to `figures/`).

## The comparison to look for

For each dataset, the paper's key question is **Reverse vs Forward XM vs K=1** on
the *biological* metrics (DEG recall, effect-r, MSE-Δ), not just E-distance:

- On Norman, **Forward XM** improved E-distance but *hurt* DEG recovery / MSE-Δ.
- Preliminary synthetic runs show **Reverse XM** improving MSE-Δ / R²-Δ (opposite
  tradeoff). Confirm on Adamson (sharper UPR effects) and Replogle (CRISPRi
  knockdowns), where perturbation effects are less subtle than Norman's CRISPRa.

Datasets differ in effect strength: Adamson (UPR, knockdown), Replogle (CRISPRi,
knockdown), Norman (CRISPRa, subtle) — a good axis to show where exploration helps.
```
