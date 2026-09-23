#!/bin/bash
#SBATCH --job-name=lozo-rankprobe-sst2-opt1.3b  # job name
#SBATCH --partition=gpu_p                # GPU partition on Sapelo2
#SBATCH --gres=gpu:A100:1                # 1x A100 (opt-1.3b fp16 fits on any Sapelo2 GPU)
#SBATCH --ntasks=1                       # single task
#SBATCH --cpus-per-task=4                # CPU cores for dataloading
#SBATCH --mem=32gb                       # host RAM
#SBATCH --time=0-08:00:00                # D-HH:MM:SS (gpu_p max is 7 days)
#SBATCH --output=logs/%x.%j.out          # stdout  (%x=job name, %j=job id)
#SBATCH --error=logs/%x.%j.err           # stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk68084@uga.edu      # <-- CHANGE to your UGA address

# =============================================================================
# LOZO :: single fine-tune of OPT-1.3B on SST2 with rank diagnostics enabled
# (--rank_probe, see large_models/rank_probe.py). Equivalent to the local
#   TASK=SST2 RANK=2 STEP_INTERVAL=50 bash lozo.sh --rank_probe ...
#
# Submit from the LOZO/large_models folder (SLURM starts the job there and
# won't create the log dir for you):
#   cd /path/to/LOZO/large_models
#   mkdir -p logs
#   sbatch ../sapelo2_scripts/run.sh
# Override the config at submit time, e.g.
#   sbatch --export=ALL,TASK=RTE,RANK=8,STEP_INTERVAL=100 ../sapelo2_scripts/run.sh
# =============================================================================
set -euo pipefail

# ---- paths (must match setup.sh) ---------------------------------------------
ENV_PREFIX="$HOME/envs/lozo"
TOKENS_FILE="$HOME/.hizoo_tokens"
export HF_HOME="/scratch/$USER/hf_cache"
OUT_ROOT="/scratch/$USER/lozo_rankprobe"
# -----------------------------------------------------------------------------

# ---- run config (overridable with sbatch --export) ----------------------------
MODEL=${MODEL:-facebook/opt-1.3b}
TASK=${TASK:-SST2}
RANK=${RANK:-2}
STEP_INTERVAL=${STEP_INTERVAL:-50}
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
# True-gradient reference layers (OPT-1.3B has 24 decoder layers); set to "" to disable
GRAD_LAYERS=${GRAD_LAYERS-'layers\.(0|6|12|18|23)\.(self_attn\.(q_proj|v_proj)|fc1)\.weight$'}
# -----------------------------------------------------------------------------

if [ ! -f lozo.sh ] || [ ! -f rank_probe.py ]; then
    echo "ERROR: submit this from the LOZO/large_models folder (no lozo.sh/rank_probe.py here)." >&2
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
# OPT is not gated, and the model/datasets are pre-cached by setup.sh.
if [ -f "$TOKENS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TOKENS_FILE"
    export HF_TOKEN
fi

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# ---- run command ------------------------------------------------------------
OUT="$OUT_ROOT/$TASK-${MODEL##*/}-r$RANK-nu$STEP_INTERVAL-s$SEED"
mkdir -p "$OUT"
echo "=== output: $OUT ==="

GRAD_ARGS=()
if [ -n "$GRAD_LAYERS" ]; then
    GRAD_ARGS=(--rank_probe_grad_layers "$GRAD_LAYERS")
fi

# lozo.sh's SEED only picks the training subset; --seed also varies the LOZO
# perturbation stream. --overwrite_output_dir: never resume, so rank_stats.csv
# always covers one uninterrupted run.
CUDA_VISIBLE_DEVICES=0 \
MODEL=$MODEL \
TASK=$TASK \
RANK=$RANK \
STEP_INTERVAL=$STEP_INTERVAL \
SEED=$SEED \
STEPS=$STEPS \
bash lozo.sh \
    --rank_probe "${GRAD_ARGS[@]}" \
    --seed "$SEED" \
    --output_dir "$OUT" \
    --result_file "$OUT/metrics.json" \
    --overwrite_output_dir

# Only rank_stats.csv / metrics.json are needed; drop the ~2.6GB checkpoint
rm -rf "$OUT"/checkpoint-*
echo "=== done :: $(date) ==="
