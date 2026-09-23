#!/bin/bash
#SBATCH --job-name=lozo-rankprobe-analyze  # job name
#SBATCH --partition=batch                # CPU partition on Sapelo2
#SBATCH --ntasks=1                       # single task
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb                       # host RAM
#SBATCH --time=0-01:00:00                # D-HH:MM:SS
#SBATCH --output=logs/%x.%j.out          # stdout
#SBATCH --error=logs/%x.%j.err           # stderr
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk68084@uga.edu      # <-- CHANGE to your UGA address

# =============================================================================
# LOZO :: plots + summary.csv for every finished rank-probe run
# (large_models/analyze_rank.py). Submit from the LOZO/large_models folder,
# after sweep_array.sh has finished (or chain it:
#   sbatch --dependency=afterany:<array job id> ../sapelo2_scripts/analyze.sh)
# =============================================================================
set -euo pipefail

# ---- paths (must match setup.sh / run.sh) ----------------------------------
ENV_PREFIX="$HOME/envs/lozo"
OUT_ROOT="/scratch/$USER/lozo_rankprobe"
KAPPA=${KAPPA:-1.0}   # constant in r*(k); see analyze_rank.py
# -----------------------------------------------------------------------------

if [ ! -f analyze_rank.py ]; then
    echo "ERROR: submit this from the LOZO/large_models folder (no analyze_rank.py here)." >&2
    exit 1
fi

module purge
module load Miniforge3/24.11.3-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

shopt -s nullglob
FILES=("$OUT_ROOT"/*/rank_stats.csv)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "ERROR: no rank_stats.csv under $OUT_ROOT" >&2
    exit 1
fi
echo "=== analyzing ${#FILES[@]} runs ==="

python analyze_rank.py "${FILES[@]}" --out "$OUT_ROOT/plots" --kappa "$KAPPA"
echo "=== plots + summary.csv in $OUT_ROOT/plots ==="
