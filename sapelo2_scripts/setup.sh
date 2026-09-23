#!/bin/bash
# =============================================================================
# LOZO :: one-time environment setup on Sapelo2 (run on a login/interactive node)
# Run from the LOZO/large_models folder of an existing clone:
#   cd /path/to/LOZO/large_models
#   bash ../sapelo2_scripts/setup.sh
# =============================================================================
set -euo pipefail

# ---- paths you may want to change --------------------------------------------
ENV_PREFIX="$HOME/envs/lozo"
TOKENS_FILE="$HOME/.hizoo_tokens"
MODEL="facebook/opt-1.3b"
# torch wheel index: cu126 runs on any CUDA 12 driver (R525+) via minor-version
# compatibility and covers A100/H100/L4. Use cu128 only if the node driver is >= 570.
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
export HF_HOME="/scratch/$USER/hf_cache"
# ------------------------------------------------------------------------------

if [ ! -f run_lozo.py ] || [ ! -f lozo.sh ] || [ ! -f ../requirements.txt ]; then
    echo "ERROR: run this from the LOZO/large_models folder (no run_lozo.py/lozo.sh here)." >&2
    exit 1
fi
mkdir -p "$HF_HOME"

echo ">>> Loading Miniforge (check 'module avail Miniforge3' if this version is gone)"
module purge
module load Miniforge3/24.11.3-0
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Create the env at the Python version the repo uses -------------------------
if [ ! -d "$ENV_PREFIX" ]; then
    echo ">>> Creating conda env at $ENV_PREFIX (python 3.9)"
    conda create -y -p "$ENV_PREFIX" python=3.9
fi
conda activate "$ENV_PREFIX"
python -V

# 2. Install torch, then requirements (+ plotting for analyze_rank.py) -----------
echo ">>> Installing torch 2.8.0 from $TORCH_INDEX"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch==2.8.0 --index-url "$TORCH_INDEX"
echo ">>> Installing ../requirements.txt"
pip install --no-cache-dir -r ../requirements.txt matplotlib

# 3. Sanity check ---------------------------------------------------------------
echo ">>> Checking library versions (CUDA is only visible on GPU nodes)"
python - <<'PY'
import torch, transformers, datasets
print("torch        ", torch.__version__)
print("transformers ", transformers.__version__)
print("datasets     ", datasets.__version__)
print("cuda avail   ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       ", torch.cuda.get_device_name(0))
PY

# 4. Pre-download model + datasets into the cache ---------------------------------
# Jobs run with HF_HUB_OFFLINE=1, so everything they need must be cached here.
echo ">>> Pre-caching $MODEL, glue/sst2 and super_glue/rte into $HF_HOME"
if [ -f "$TOKENS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TOKENS_FILE"
    export HF_TOKEN
fi
MODEL="$MODEL" python - <<'PY'
import os
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset

model = os.environ["MODEL"]
snapshot_download(model)   # ~2.6GB for opt-1.3b
AutoConfig.from_pretrained(model)
AutoTokenizer.from_pretrained(model, use_fast=False)
for args in [("glue", "sst2"), ("super_glue", "rte")]:
    d = load_dataset(*args)
    print(args, {k: len(v) for k, v in d.items()})
print("cache warm :: OK")
PY

echo
echo "================================================================"
echo " Setup complete."
echo "   code : $(pwd)"
echo "   env  : $ENV_PREFIX"
echo "   cache: $HF_HOME"
echo " Next, from this same folder:"
echo "   mkdir -p logs"
echo "   sbatch ../sapelo2_scripts/run.sh            # one run (smoke test)"
echo "   sbatch ../sapelo2_scripts/sweep_array.sh    # full rank-probe sweep"
echo "================================================================"
