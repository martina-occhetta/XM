#!/bin/bash
#SBATCH -J pertxm_download
#SBATCH -A pilot
#SBATCH -p compute
#SBATCH -n 2
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=12G
#SBATCH -o logs/01_download.o%j
# Apocrita (Slurm). Downloads + caches the real Perturb-seq datasets ONCE into
# $DATA_DIR. --mem-per-cpu is PER core, so total RAM = 4 x 12G = 48G (Perturb-seq
# objects are large; bump if a loader OOMs). Needs network access.
#
# EDIT: -A account, -p partition, DATASETS, DATA_DIR.
set -euo pipefail

module load python/3.12 2>/dev/null || true
ENV_DIR="${ENV_DIR:-.pertxm_env}"
source "$ENV_DIR/bin/activate"

export DATA_DIR="${DATA_DIR:-$PWD/data/perturbseq}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"; mkdir -p "$NUMBA_CACHE_DIR" "$DATA_DIR" logs
# pertpy/scanpy also honor these for cache location:
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DIR/.cache}"

DATASETS="${DATASETS:-adamson_2016 norman_2019}"

python "/data/SBCS-BessantLab/martina/pert_xm/XM/experiments/perturbation_xm/jobs/download_datasets.py" --datasets $DATASETS --cache-dir "$DATA_DIR"
echo "[data cached in] $DATA_DIR"
