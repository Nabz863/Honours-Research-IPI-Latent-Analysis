#!/bin/bash
#SBATCH --job-name=ipi_baseline_internvl
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/baseline_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/baseline_%j.err

# ============================================================
# Phase 1: Clean-condition baseline collection
# Wits HPC (SLURM) job script — mscluster bigbatch partition
#
# Hardware: Quadro RTX 8000 (49GB VRAM), CUDA 13.3
# Submit with: sbatch hpc_baseline.sh
# ============================================================

set -euo pipefail

mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

# ---- System CUDA -----------------------------------------------
export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ---- Environment -----------------------------------------------
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate

# ---- HuggingFace cache (weights in /datasets, not home dir) ----
export HF_HOME=/datasets/nmahomed/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"

# ---- Run -------------------------------------------------------
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

python baselines/collect_baselines.py \
    --model      internvl8b \
    --n_samples  100 \
    --output_dir baselines/internvl8b_clean \
    --modality   multimodal \
    --token_aggregation last \
    --hpc

echo "Job finished at $(date)"