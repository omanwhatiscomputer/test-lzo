#!/bin/bash
# =============================================================================
# LOZO :: one-time environment setup on Sapelo2 (run on a login/interactive node)
# Run from the root of an existing LOZO clone:
#   cd /path/to/LOZO
#   bash scripts/setup.sh
# =============================================================================
set -euo pipefail

# ---- paths you may want to change --------------------------------------------
ENV_PREFIX="$HOME/envs/lozo"
TOKENS_FILE="$HOME/.hizoo_tokens"
MODEL="facebook/opt-2.7b"
export HF_HOME="/scratch/$USER/hf_cache"
# ------------------------------------------------------------------------------

if [ ! -f requirements.txt ] || [ ! -f large_models/lozo.sh ]; then
    echo "ERROR: run this from the LOZO repo root (no requirements.txt/large_models/lozo.sh here)." >&2
    exit 1
fi
mkdir -p "$HF_HOME"

echo ">>> Loading Miniforge (check 'module avail Miniforge3' if this version is gone)"
module purge
module load Miniforge3/24.11.3-0
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Create the env at the Python version the repo was tested on ------------------
if [ ! -d "$ENV_PREFIX" ]; then
    echo ">>> Creating conda env at $ENV_PREFIX (python 3.9.7)"
    conda create -y -p "$ENV_PREFIX" python=3.9.7
fi
conda activate "$ENV_PREFIX"
python -V

# 2. Install requirements --------------------------------------------------------
# torch first from the CUDA 12.8 index (covers A100/H100/L4 and Blackwell), then the rest
echo ">>> Installing torch (cu128) and requirements.txt"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir -r requirements.txt

# 3. Sanity check ---------------------------------------------------------------
echo ">>> Checking library versions (CUDA is only visible on GPU nodes)"
python - <<'PY'
import torch, transformers, accelerate, datasets, numpy
print("torch        ", torch.__version__)
print("transformers ", transformers.__version__)
print("accelerate   ", accelerate.__version__)
print("datasets     ", datasets.__version__)
print("numpy        ", numpy.__version__)
print("cuda avail   ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       ", torch.cuda.get_device_name(0))
PY

# 4. Pre-download model + dataset into the cache ---------------------------------
echo ">>> Pre-caching $MODEL and glue/sst2 into $HF_HOME"
export HF_DATASETS_TRUST_REMOTE_CODE=1
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
# Download the weights without loading 2.7B parameters into login-node RAM.
snapshot_download(model)   # ~5GB download
AutoConfig.from_pretrained(model)
AutoTokenizer.from_pretrained(model, use_fast=False)
d = load_dataset("glue", "sst2")
print({k: len(v) for k, v in d.items()})
print("cache warm :: OK")
PY

echo
echo "================================================================"
echo " Setup complete."
echo "   code : $(pwd)"
echo "   env  : $ENV_PREFIX"
echo "   cache: $HF_HOME"
echo " Next:"
echo "   cd large_models && mkdir -p logs && sbatch ../scripts/run.sh"
echo "================================================================"
