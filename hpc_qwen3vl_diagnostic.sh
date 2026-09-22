#!/bin/bash
#SBATCH --job-name=qwen3vl_diagnostic
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/qwen3vl_diagnostic_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/qwen3vl_diagnostic_%j.err

# Small, standalone diagnostic -- 5 tasks, 2 conditions each, no
# checkpointing/resume needed (cheap enough to just rerun if it fails).
# --time=01:00:00 is generous for 10 episodes; this is NOT the full
# matrix and shouldn't be confused with it in squeue.

set -euo pipefail

mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate

export HF_HOME=/datasets/nmahomed/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

# Deliberately NOT --hpc_offline/HF_HUB_OFFLINE=1 here, unlike the main
# matrix -- this is Qwen3-VL-8B's FIRST load, it needs to actually reach
# huggingface.co to download weights. If this errors on a network/auth
# issue, that's the real, expected first failure mode to debug, not a
# code bug (see model_loader.py's own docstring on this exact point).
python experiments/qwen3vl_diagnostic.py

echo "Job finished at $(date)"
