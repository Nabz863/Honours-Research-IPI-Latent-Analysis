#!/bin/bash
#SBATCH --job-name=ipi_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# ============================================================
# Phase 1: Clean-condition baseline collection
# Wits HPC (SLURM) job script
#
# Submit with:
#   sbatch hpc_baseline.sh
#
# To collect text-only baselines, change --modality to text.
# To use the 13B model, change --model to llava13b and add
# --quantise if VRAM < 40GB (sets load_in_4bit=True — flag
# results as potentially precision-affected).
# ============================================================

set -euo pipefail

mkdir -p logs

# ---- Environment -----------------------------------------------
module load python/3.12 cuda/12.1 2>/dev/null || true

# Activate your virtualenv / conda env if needed:
# source ~/envs/ipi/bin/activate

# ---- HuggingFace cache (use scratch to avoid home quota) -------
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ---- Run -------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

python baselines/collect_baselines.py \
    --model      llava7b \
    --n_samples  100 \
    --output_dir baselines/llava7b_clean \
    --modality   multimodal \
    --token_aggregation last \
    --hpc

echo "Job finished at $(date)"
