#!/bin/bash
#SBATCH -J pertxm_ablation
#SBATCH -A pilot_andrena
#SBATCH -p andrena
#SBATCH -n 16
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=8G
#SBATCH -o logs/03_ablation.o%j
# Apocrita (Slurm). The heavy run: K in {1,2,4,8} x steps {2,4,10} x 5 seeds,
# with bootstrap CIs + the ablation figure, on a REAL cached dataset. This is the
# paper's main figure. Reads the cache from 01 (no network).
#
# GPU (recommended if you drop the subset knobs / scale up): switch -p to your GPU
# partition, add  #SBATCH --gres=gpu:1 , and set XM_DEVICE=cuda below. At the
# default subset scale this fits comfortably on CPU within the 24h budget.
set -euo pipefail

module load python/3.12 2>/dev/null || true
ENV_DIR="${ENV_DIR:-.pertxm_env}"
source "$ENV_DIR/bin/activate"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
EXP_DIR="$REPO_ROOT/experiments/perturbation_xm"
export DATA_DIR="${DATA_DIR:-$PWD/data/perturbseq}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"; mkdir -p "$NUMBA_CACHE_DIR" logs results figures
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# export XM_DEVICE=cuda   # uncomment on a GPU node

DATASET="${DATASET:-replogle_2022_k562}"
SPACE="${SPACE:-nbvae}"

python "$EXP_DIR/scldm_ablation.py" \
    --dataset "$DATASET" --space "$SPACE" \
    --ks 1 2 4 8 --steps 2 4 10 --seeds 0 1 2 3 4 --updates 3000 \
    --n-hvg 2000 --max-perts 20 --max-cells-per-cond 400 \
    --cache-dir "$DATA_DIR" --strict-dataset --plot \
    --out-dir results

cp -f "results/scldm_ablation_${DATASET}_${SPACE}.png" "figures/" 2>/dev/null || true
echo "[done] results/scldm_ablation_${DATASET}_${SPACE}.{json,png}"
