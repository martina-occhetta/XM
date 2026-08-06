# Cluster jobs (Slurm / Apocrita)

Slurm scripts to run the perturbation-XM experiments on a cluster, downloading
each real dataset **once** and reusing the cache for all compute jobs. Styled for
Apocrita (`-A pilot_andrena -p andrena`); edit the `#SBATCH` headers for your
account/partition.

## Pipeline

| Stage | Script | What it does | Network? | GPU? |
| --- | --- | --- | --- | --- |
| 0 | `00_setup_env.sh` | build a Python **3.12** venv, `pip install -e ..[datasets]` | yes | no |
| 1 | `01_download_data.sh` | download + cache datasets into `$DATA_DIR` (via `download_datasets.py`) | yes | no |
| 2 | `02_bio_eval.sh` | biological eval (E-distance + DEG + pathway), K=1 vs XM | no | optional |
| 3 | `03_ablation.sh` | K×steps ablation + bootstrap CIs + figure (paper's main figure) | no | optional |

Submit the whole chain (each stage waits for the previous):

```bash
cd /path/to/your/project            # where logs/ data/ results/ figures/ should live
bash experiments/perturbation_xm/jobs/submit_all.sh
# or one at a time:
sbatch experiments/perturbation_xm/jobs/00_setup_env.sh
```

## Before you submit — edit these

1. **`#SBATCH -A` / `-p`** in every script → your account / partition.
2. **Python module**: scripts `module load python/3.12`. bio-perturbations needs
   **≥3.12** (not 3.11). Check `module avail python` and adjust.
3. **torch build**: `00_setup_env.sh` installs the default PyPI wheel (CUDA-enabled
   on Linux — works on GPU nodes). For a **CPU-only** cluster, change it to
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
4. **Datasets**: set `DATASETS` in `01` (default `replogle_2022_k562 norman_2019`).
   Compute jobs pick one via `DATASET=` (default `replogle_2022_k562`).

## Config knobs (env vars, override at submit time)

```bash
DATASET=norman_2019 SPACE=nbvae sbatch experiments/perturbation_xm/jobs/03_ablation.sh
```

- `ENV_DIR` (default `.pertxm_env`) — venv location.
- `DATA_DIR` (default `./data/perturbseq`) — dataset cache (share across jobs).
- `DATASET` (default `replogle_2022_k562`), `SPACE` (default `nbvae`).
- `GENE_SETS` (job 2) — path to a `{name:[genes]}` JSON for pathway metrics
  (e.g. MSigDB Hallmark, or each perturbation's DEGs). Omitted → pathway metrics off.
- `XM_DEVICE` — `cpu` / `cuda`. Auto-detects CUDA; the scripts are device-aware.

## GPU

At the default subset scale (`--n-hvg 2000 --max-perts 20 --max-cells-per-cond 400`)
these fit on CPU within the time budgets. Use a GPU when you **scale up** (drop the
subset knobs, full genes/cells, more seeds/updates, or a real scLDM network):
switch `-p` to your GPU partition, add `#SBATCH --gres=gpu:1`, and uncomment
`export XM_DEVICE=cuda` in jobs 2/3. Note the **bio-perturbations scoring**
(DESeq2/PCA/E-distance) is CPU/numpy regardless — GPU only speeds training.

## Outputs

- `results/bioeval_<dataset>_<space>.json` — per-model biological metrics.
- `results/scldm_ablation_<dataset>_<space>.json` + `.png` (also copied to `figures/`)
  — the K×steps grid with CIs and the paper figure.
- `logs/*.o<jobid>` — stdout per job.

Drop the figure into `paper/figs/ablation.png` and transcribe the JSON numbers
into `paper/main.tex` (`tab:main`) — see `paper/README.md`.
