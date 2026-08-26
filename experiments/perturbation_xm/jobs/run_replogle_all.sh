#!/bin/bash
# ---------------------------------------------------------------------------
# STEP 1 of 2 -- run ALL flow experiments on Replogle (K562 + RPE1)
#
# Genome-scale CRISPRi Perturb-seq. ALL perturbations, matched latent-64,
# NB-VAE (the scLDM setting), 3 seeds, K in {1,4}, forward AND reverse XM.
# Bio-eval (E-distance + DEG + pathway) for each; ablation is optional (below).
#
# Submit from the repo root on Apocrita:   bash experiments/perturbation_xm/jobs/run_replogle_all.sh
#
# PREREQS (do these once, before submitting):
#   1. Sync the latest code bundle (must contain commit fcaee77 -- adds
#      --generator, --min-cells-per-pert, and the VAE NaN fix). Verify with:
#         git -C ~/XM log --oneline -1     # expect fcaee77 or later
#   2. Activate/point at the venv that has bio-perturbations installed
#      (this script uses BPVENV below; edit if yours lives elsewhere).
#   3. RPE1 must be cached. This script submits a download job first and makes
#      the run jobs depend on it. If RPE1 is already cached it's a fast no-op.
#      NOTE: confirm the exact RPE1 id in YOUR bio-perturbations build:
#         python -c "from bio_perturbations.datasets import list_datasets; print(list_datasets())"
#      and set RPE1_ID below if it differs from replogle_2022_rpe1.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/XM}"
EXP_DIR="$REPO_ROOT/experiments/perturbation_xm"
BPVENV="${BPVENV:-/workspace/bpvenv}"            # venv with bio-perturbations
CACHE="${CACHE:-$REPO_ROOT/data/perturbseq}"     # dataset cache (reused)
RESULTS="${RESULTS:-$REPO_ROOT/results}"
LOGS="${LOGS:-$REPO_ROOT/logs}"
mkdir -p "$RESULTS" "$LOGS" "$CACHE"

K562_ID="${K562_ID:-replogle_2022_k562}"
RPE1_ID="${RPE1_ID:-replogle_2022_rpe1}"

# Genome-scale settings (thousands of perts x pseudobulk DESeq2 -> heavy RAM/time)
ACCT="pilot"; PART="highmem"          # highmem QOS allows mem>=128G
NCPU=8; MEMPC="24G"                    # 8 x 24G = 192G
WALL="48:0:0"
COMMON="--space nbvae --latent-dim 64 --vae-epochs 200 --n-hvg 2000 \
--max-perts 0 --min-cells-per-pert 50 --max-cells-per-cond 400 \
--updates 5000 --n_gen 300 --n_steps 10 --ks 1 4 --seeds 0 1 2 --strict-dataset"
# NOTE: --strict-dataset makes the run FAIL rather than silently fall back to
# synthetic if the real dataset can't load. Drop it only if you want a fallback.

RUN_ABLATION="${RUN_ABLATION:-0}"      # set to 1 to also submit the K x steps grid

# ---- 0. ensure RPE1 is downloaded (K562 assumed already cached from prior runs)
dl=$(sbatch --parsable -A "$ACCT" -p "$PART" -n 4 --mem-per-cpu="$MEMPC" -t 8:0:0 \
  -J rep_dl -o "$LOGS/rep_dl.o%j" --wrap \
  "source $BPVENV/bin/activate; cd $REPO_ROOT; export NUMBA_CACHE_DIR=\${TMPDIR:-/tmp}/numba; python $EXP_DIR/jobs/download_datasets.py --datasets $RPE1_ID --cache-dir $CACHE")
echo "submitted RPE1 download: job $dl"

# ---- 1. bio-eval: for each dataset x direction, one job (all K/seeds inside)
submit_bioeval () {
  local ds="$1" dir="$2"
  local out="$RESULTS/bioeval_${ds}_nbvae_L64_all_${dir}.json"
  local jid
  jid=$(sbatch --parsable -A "$ACCT" -p "$PART" -n "$NCPU" --mem-per-cpu="$MEMPC" -t "$WALL" \
    --dependency=afterok:"$dl" -J "be_${ds}_${dir}" -o "$LOGS/be_${ds}_${dir}.o%j" --wrap \
    "source $BPVENV/bin/activate; cd $REPO_ROOT; export OMP_NUM_THREADS=$NCPU NUMBA_CACHE_DIR=\${TMPDIR:-/tmp}/numba; python $EXP_DIR/scldm_bio_eval.py --dataset $ds --cache-dir $CACHE --xm-direction $dir $COMMON --out $out")
  echo "  bio-eval $ds/$dir -> job $jid -> $out"
}

# ---- 2. (optional) ablation: K x steps grid with bootstrap CIs + figure
submit_ablation () {
  local ds="$1" dir="$2"
  local jid
  jid=$(sbatch --parsable -A "$ACCT" -p "$PART" -n "$NCPU" --mem-per-cpu="$MEMPC" -t "$WALL" \
    --dependency=afterok:"$dl" -J "ab_${ds}_${dir}" -o "$LOGS/ab_${ds}_${dir}.o%j" --wrap \
    "source $BPVENV/bin/activate; cd $REPO_ROOT; export OMP_NUM_THREADS=$NCPU NUMBA_CACHE_DIR=\${TMPDIR:-/tmp}/numba; python $EXP_DIR/scldm_ablation.py --dataset $ds --cache-dir $CACHE --xm-direction $dir $COMMON --out-dir $RESULTS")
  echo "  ablation $ds/$dir -> job $jid"
}

for ds in "$K562_ID" "$RPE1_ID"; do
  for dir in forward reverse; do
    submit_bioeval "$ds" "$dir"
    [ "$RUN_ABLATION" = "1" ] && submit_ablation "$ds" "$dir"
  done
done

echo
echo "Queued. Watch with:  squeue -u \$USER"
echo "Results land in:     $RESULTS/bioeval_replogle_2022_{k562,rpe1}_nbvae_L64_all_{forward,reverse}.json"
