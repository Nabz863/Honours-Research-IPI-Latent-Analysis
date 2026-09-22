#!/bin/bash
#SBATCH --job-name=qwen3vl_nan_repro
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster44,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster69,mscluster70,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/qwen3vl_nan_repro_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/qwen3vl_nan_repro_%j.err

# bigbatch, not biggpu -- this should queue in minutes, not days, for a
# ~2-minute check. CUDA_LAUNCH_BLOCKING=1 trades speed for an accurate,
# synchronous traceback pointing at the real originating kernel instead
# of wherever the async error happens to surface.

set -uo pipefail
mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
python3 experiments/qwen3vl_nan_repro.py
echo "Finished at $(date)"
