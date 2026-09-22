#!/bin/bash
#SBATCH --job-name=fetch_llama
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=03:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/fetch_llama_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/fetch_llama_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache
unset HF_HUB_OFFLINE
python - <<'PY'
from huggingface_hub import snapshot_download
print("cached at", snapshot_download("meta-llama/Llama-3.1-8B-Instruct"))
PY
