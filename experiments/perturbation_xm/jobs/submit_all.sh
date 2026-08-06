#!/bin/bash
# Submit the whole pipeline with Slurm dependencies: setup -> download -> {bioeval, ablation}.
# Each stage waits for the previous to finish OK. Run from the directory where you
# want logs/, data/, results/, figures/ to live (e.g. your project root on the cluster).
#
#   bash experiments/perturbation_xm/jobs/submit_all.sh
#
# Override defaults via env, e.g.:
#   DATASET=norman_2019 SPACE=nbvae bash .../submit_all.sh
set -euo pipefail
JOBS="$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs

setup=$(sbatch --parsable "$JOBS/00_setup_env.sh")
echo "setup:    $setup"
dl=$(sbatch --parsable --dependency=afterok:$setup "$JOBS/01_download_data.sh")
echo "download: $dl"
be=$(sbatch --parsable --dependency=afterok:$dl "$JOBS/02_bio_eval.sh")
echo "bioeval:  $be"
ab=$(sbatch --parsable --dependency=afterok:$dl "$JOBS/03_ablation.sh")
echo "ablation: $ab"
echo "submitted. watch with:  squeue -u \$USER"
