#!/bin/bash
#SBATCH --job-name=lozo-rankprobe-sweep  # job name
#SBATCH --partition=gpu_p                # GPU partition on Sapelo2
#SBATCH --gres=gpu:A100:1                # 1 GPU per array task
#SBATCH --ntasks=1                       # single task
#SBATCH --cpus-per-task=4                # CPU cores for dataloading
#SBATCH --mem=32gb                       # host RAM
#SBATCH --time=0-08:00:00                # D-HH:MM:SS, per array task
#SBATCH --array=0-47%8                   # 2 tasks x 4 ranks x 2 nu x 3 seeds; at most 8 GPUs at once
#SBATCH --output=logs/%x.%A_%a.out       # stdout  (%A=array job id, %a=task index)
#SBATCH --error=logs/%x.%A_%a.err        # stderr
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=mk68084@uga.edu      # <-- CHANGE to your UGA address

# =============================================================================
# LOZO :: rank-diagnostic sweep as a SLURM array (Sapelo2 version of
# large_models/rank_sweep.sh). Each array task is one (task, r, nu, seed) run
# of run.sh; results land in /scratch/$USER/lozo_rankprobe/<config>/.
#
# Submit from the LOZO/large_models folder:
#   cd /path/to/LOZO/large_models
#   mkdir -p logs
#   sbatch ../sapelo2_scripts/sweep_array.sh
# Re-run only some configs by index, e.g.  sbatch --array=5,17 ../sapelo2_scripts/sweep_array.sh
# If you change the lists below, set --array=0-(N-1) with N = product of their lengths.
# When all tasks finish:  sbatch ../sapelo2_scripts/analyze.sh
# =============================================================================
set -euo pipefail

TASKS=(SST2 RTE)
RANKS=(2 4 8 16)
INTERVALS=(50 100)
SEEDS=(0 1 2)

i=$SLURM_ARRAY_TASK_ID
N=$(( ${#TASKS[@]} * ${#RANKS[@]} * ${#INTERVALS[@]} * ${#SEEDS[@]} ))
if [ "$i" -ge "$N" ]; then
    echo "ERROR: array index $i >= $N configs; fix --array." >&2
    exit 1
fi
export SEED=${SEEDS[$(( i % ${#SEEDS[@]} ))]};                        i=$(( i / ${#SEEDS[@]} ))
export STEP_INTERVAL=${INTERVALS[$(( i % ${#INTERVALS[@]} ))]};       i=$(( i / ${#INTERVALS[@]} ))
export RANK=${RANKS[$(( i % ${#RANKS[@]} ))]};                        i=$(( i / ${#RANKS[@]} ))
export TASK=${TASKS[$i]}

echo "=== array task $SLURM_ARRAY_TASK_ID :: TASK=$TASK RANK=$RANK STEP_INTERVAL=$STEP_INTERVAL SEED=$SEED ==="

# sbatch runs a copy of this script from the spool dir, so find run.sh via the submit dir
exec bash "$SLURM_SUBMIT_DIR/../sapelo2_scripts/run.sh"
