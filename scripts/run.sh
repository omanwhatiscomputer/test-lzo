#!/bin/bash
#SBATCH --job-name=lozo-sst2-opt2.7b-r4-2 # job name
#SBATCH --partition=gpu_p                # GPU partition on Sapelo2
#SBATCH --gres=gpu:A100:1                # 1x A100 (LOZO on OPT-2.7B in fp16 needs well under 16GB)
#SBATCH --ntasks=1                       # single task
#SBATCH --cpus-per-task=5                # CPU cores for dataloading
#SBATCH --mem=32gb                       # host RAM
#SBATCH --time=0-12:00:00                # D-HH:MM:SS (gpu_p max is 7 days)
#SBATCH --output=logs/%x.%j.out          # stdout  (%x=job name, %j=job id)
#SBATCH --error=logs/%x.%j.err           # stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk68084@uga.edu      # <-- CHANGE to your UGA address

# =============================================================================
# LOZO :: full-parameter LOZO fine-tune of OPT-2.7B on SST2
# Rank schedule: r=4 for the first 10000 steps, then r=2 for the next 10000.
# Equivalent to the local command:
#   MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 \
#   RANK_SCHEDULE=4:10000,2 STEP_INTERVAL=50 bash lozo.sh
#
# Submit from the LOZO/large_models folder (SLURM starts the job there and
# won't create the log dir for you):
#   cd /path/to/LOZO/large_models
#   mkdir -p logs
#   sbatch ../scripts/run.sh
# =============================================================================
set -euo pipefail

# ---- paths (must match setup.sh) ---------------------------------------------
ENV_PREFIX="$HOME/envs/lozo"
TOKENS_FILE="$HOME/.hizoo_tokens"
export HF_HOME="/scratch/$USER/hf_cache"
# -----------------------------------------------------------------------------

if [ ! -f lozo.sh ]; then
    echo "ERROR: submit this from the LOZO/large_models folder (no lozo.sh here)." >&2
    exit 1
fi

echo "=== job $SLURM_JOB_ID on $(hostname) :: $(date) ==="
echo "=== running in $(pwd) ==="
nvidia-smi

# ---- environment -------------------------------------------------------------
module purge
module load Miniforge3/24.11.3-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# ---- tokens (optional) ------------------------------------------------------
# OPT-2.7B is not gated, so HF_TOKEN is only needed to avoid rate limits.
if [ -f "$TOKENS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TOKENS_FILE"
    export HF_TOKEN
fi

export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# ---- run command ------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 \
MODEL=facebook/opt-2.7b \
TASK=SST2 \
MODE=ft \
LR=1e-7 \
EPS=1e-3 \
BS=16 \
TRAIN=1000 \
DEV=500 \
EVAL=1000 \
STEPS=20000 \
EVAL_STEPS=4000 \
RANK_SCHEDULE=4:10000,2 \
STEP_INTERVAL=50 \
SEED=0 \
bash lozo.sh

echo "=== finished $(date) ==="
