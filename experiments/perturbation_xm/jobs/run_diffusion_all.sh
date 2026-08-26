#!/bin/bash
# ---------------------------------------------------------------------------
# STEP 2 of 2 -- run the DIFFUSION backbone on ALL datasets
#
# Same harness, latent diffusion generator (--generator diffusion), which is
# the actual scLDM formulation: x0-prediction, cosine schedule, DDIM sampling.
# Datasets: adamson_2016, norman_2019, replogle_2022_k562, replogle_2022_rpe1.
# ALL perturbations, matched latent-64, NB-VAE, 3 seeds, K in {1,4}, fwd + rev.
#
# Submit from the repo root:  bash experiments/perturbation_xm/jobs/run_diffusion_all.sh
#
# PREREQS: same as run_replogle_all.sh -- code bundle must be >= fcaee77
# (that commit is what adds --generator), all four datasets cached, venv ready.
# Run STEP 1 (or at least the RPE1 download) first so RPE1 is on disk.
#
# EXPECTATION: diffusion is very likely to reproduce the flow result --
# XM moves the distribution metric (E-distance) but not the biology
# (DEG recovery / effect-size correlation). That's the point of the paper:
# the finding is about the exploration objective, not the specific backbone.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/XM}"
EXP_DIR="$REPO_ROOT/experiments/perturbation_xm"
BPVENV="${BPVENV:-/workspace/bpvenv}"
CACHE="${CACHE:-$REPO_ROOT/data/perturbseq}"
RESULTS="${RESULTS:-$REPO_ROOT/results}"
LOGS="${LOGS:-$REPO_ROOT/logs}"
mkdir -p "$RESULTS" "$LOGS"

# Shared knobs. --generator diffusion + --n-steps 20 (DDIM needs a few more
# steps than the Euler flow to denoise cleanly).
COMMON="--space nbvae --latent-dim 64 --vae-epochs 200 --n-hvg 2000 \
--max-perts 0 --min-cells-per-pert 50 --max-cells-per-cond 400 \
--updates 5000 --n_gen 300 --generator diffusion --n_steps 20 \
--ks 1 4 --seeds 0 1 2 --strict-dataset"

# Per-dataset resources: Adamson/Norman are small enough for the compute
# partition; the two Replogle sets are genome-scale and need highmem + 48h.
res_for () {  # echoes: ACCT PART NCPU MEMPC WALL
  case "$1" in
    replogle_2022_k562|replogle_2022_rpe1) echo "pilot highmem 8 24G 48:0:0" ;;
    *)                                     echo "pilot compute 8 8G 16:0:0" ;;
  esac
}

submit_diff () {
  local ds="$1" dir="$2"
  read -r ACCT PART NCPU MEMPC WALL < <(res_for "$ds")
  local out="$RESULTS/bioeval_${ds}_nbvae_L64_all_diffusion_${dir}.json"
  local jid
  jid=$(sbatch --parsable -A "$ACCT" -p "$PART" -n "$NCPU" --mem-per-cpu="$MEMPC" -t "$WALL" \
    -J "df_${ds}_${dir}" -o "$LOGS/df_${ds}_${dir}.o%j" --wrap \
    "source $BPVENV/bin/activate; cd $REPO_ROOT; export OMP_NUM_THREADS=$NCPU NUMBA_CACHE_DIR=\${TMPDIR:-/tmp}/numba; python $EXP_DIR/scldm_bio_eval.py --dataset $ds --cache-dir $CACHE --xm-direction $dir $COMMON --out $out")
  echo "  diffusion $ds/$dir [$PART] -> job $jid -> $out"
}

for ds in adamson_2016 norman_2019 replogle_2022_k562 replogle_2022_rpe1; do
  for dir in forward reverse; do
    submit_diff "$ds" "$dir"
  done
done

echo
echo "Queued. Watch with:  squeue -u \$USER"
echo "Results:  $RESULTS/bioeval_<dataset>_nbvae_L64_all_diffusion_{forward,reverse}.json"
echo "Compare each against its flow twin (same name without _diffusion) to see"
echo "whether the backbone changes the XM tradeoff -- expected: it does not."
