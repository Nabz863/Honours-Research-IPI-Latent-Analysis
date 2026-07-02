#!/bin/bash
#SBATCH --job-name=ipi_baseline
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# ============================================================
# Phase 1: Clean-condition baseline collection
# Wits HPC (SLURM) job script — mscluster biggpu partition
#
# Hardware: Quadro RTX 8000 (49GB VRAM), CUDA 13.2
# Note: --gres=gpu:1 omitted intentionally — GRES not yet
# configured on the post-attack rebuild (July 2026). GPUs
# are physically present and visible to PyTorch via CUDA.
# Re-add --gres=gpu:1 once MSS confirms GRES registration.
#
# Submit with:
#   sbatch hpc_baseline.sh
#
# To collect text-only baselines, change --modality to text.
# To use the 13B model, change --model to llava13b.
# ============================================================

set -euo pipefail

mkdir -p logs

# ---- Environment -----------------------------------------------
source ~/Honours-Research-IPI-Latent-Analysis/venv/bin/activate

# ---- HuggingFace cache (keep off home directory quota) ---------
export HF_HOME=/home-mscluster/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"

# ---- Run -------------------------------------------------------
cd ~/Honours-Research-IPI-Latent-Analysis

python baselines/collect_baselines.py \
    --model      llava7b \
    --n_samples  100 \
    --output_dir baselines/llava7b_clean \
    --modality   multimodal \
    --token_aggregation last \
    --hpc

echo "Job finished at $(date)"
