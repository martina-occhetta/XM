#!/bin/bash
#SBATCH -J pertxm_bioeval
#SBATCH -A pilot_andrena
#SBATCH -p andrena
#SBATCH -n 8
#SBATCH -t 12:0:0
#SBATCH --mem-per-cpu=8G
#SBATCH -o logs/02_bioeval.o%j
# Apocrita (Slurm). Biological evaluation (E-distance + DEG + pathway) of the
# XM-wrapped generator vs baselines on a REAL cached dataset. Reads the cache from
# 01_download_data.sh (no network needed here).
#
# GPU: to run on a GPU node, switch -p to your GPU partition and add e.g.
#   #SBATCH --gres=gpu:1
# then set XM_DEVICE=cuda below. At the default subset scale CPU is fine.
set -euo pipefail

module load python/3.12 2>/dev/null || true
ENV_DIR="${ENV_DIR:-.pertxm_env}"
source "$ENV_DIR/bin/activate"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
EXP_DIR="$REPO_ROOT/experiments/perturbation_xm"
export DATA_DIR="${DATA_DIR:-$PWD/data/perturbseq}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"; mkdir -p "$NUMBA_CACHE_DIR" logs results
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# export XM_DEVICE=cuda   # uncomment on a GPU node

DATASET="${DATASET:-replogle_2022_k562}"
SPACE="${SPACE:-nbvae}"          # count decoder for real counts
GENE_SETS="${GENE_SETS:-}"       # optional path to {name:[genes]} JSON for pathway metrics
GS_ARG=""; [ -n "$GENE_SETS" ] && GS_ARG="--gene-sets $GENE_SETS"

DSUF=""; [ "$XM_DIRECTION" != "forward" ] && DSUF="_${XM_DIRECTION}"
python "$EXP_DIR/scldm_bio_eval.py" \
    --dataset "$DATASET" --space "$SPACE" \
    --n-hvg 2000 --max-perts 20 --max-cells-per-cond 400 \
    --seeds 0 1 2 --ks 1 4 --updates 5000 \
    --cache-dir "$DATA_DIR" --strict-dataset $GS_ARG \
    --out "results/bioeval_${DATASET}_${SPACE}${DSUF}.json"

echo "[done] results/bioeval_${DATASET}_${SPACE}.json"
