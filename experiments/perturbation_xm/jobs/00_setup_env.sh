#!/bin/bash
#SBATCH -J pertxm_setup
#SBATCH -A pilot_andrena
#SBATCH -p andrena
#SBATCH -n 4
#SBATCH -t 2:0:0
#SBATCH --mem-per-cpu=8G
#SBATCH -o logs/00_setup.o%j
# Apocrita (Slurm). Builds the Python venv and installs perturbation-xm + the
# bio-perturbations evaluation stack. Run this on a LOGIN/COMPUTE node that has
# network access (compute nodes on many clusters do; if not, run on a login node).
#
# EDIT for your cluster: -A account, -p partition, and the python module below.
# IMPORTANT: bio-perturbations requires Python >= 3.12 (not 3.11) -- load a 3.12
# module. Check `module avail python`.
set -euo pipefail

module load python/3.12 2>/dev/null || module load python/3.12.3 2>/dev/null || true
python --version

ENV_DIR="${ENV_DIR:-.pertxm_env}"           # override: ENV_DIR=/path sbatch 00_setup_env.sh
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"  # repo root (contains experiments/)
EXP_DIR="$REPO_ROOT/experiments/perturbation_xm"

mkdir -p logs data figures

# Fresh venv
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip

# torch: the default PyPI Linux wheel is CUDA-enabled (works on GPU nodes).
# For a CPU-only cluster, instead:  pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch

# The experiment package + biological evaluation + real-data (pertpy) loaders.
# The [datasets] extra pulls bio-perturbations[prep,datasets] from GitHub.
pip install -e "${EXP_DIR}[datasets]"

python -c "import torch, anndata, bio_perturbations, pertpy; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('bio_perturbations', bio_perturbations.__version__)"
echo "[setup complete] venv: $ENV_DIR"
