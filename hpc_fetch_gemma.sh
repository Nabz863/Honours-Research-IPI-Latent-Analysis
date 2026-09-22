#!/bin/bash
#SBATCH --job-name=fetch_gemma
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=03:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/fetch_gemma_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/fetch_gemma_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache
unset HF_HUB_OFFLINE
python - <<'PY'
from huggingface_hub import snapshot_download
print("cached at", snapshot_download("google/gemma-4-12B-it"))
PY
python - <<'PY'
from transformers import AutoProcessor, AutoConfig
p = AutoProcessor.from_pretrained("google/gemma-4-12B-it")
ip = getattr(p, "image_processor", p)
print("image_processor:", type(ip).__name__)
for a in ("image_grid_pinpoints","crop_size","merge_size","patch_size",
          "min_patches","max_patches","size","image_mean"):
    print(f"  {a}:", getattr(ip, a, None))
c = AutoConfig.from_pretrained("google/gemma-4-12B-it")
t = getattr(c, "text_config", c)
print("hidden:", getattr(t,"hidden_size",None), "| layers:", getattr(t,"num_hidden_layers",None))
PY
