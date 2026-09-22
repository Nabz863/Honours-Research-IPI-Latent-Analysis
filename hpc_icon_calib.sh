#!/bin/bash
#SBATCH --job-name=icon_calib
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster111,mscluster107
#SBATCH --nodes=1 --cpus-per-task=8 --mem=48G --time=04:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/icon_calib_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/icon_calib_%j.err
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache TRANSFORMERS_CACHE=$HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python experiments/calibrate_icon.py --model qwen3vl --n_tasks 60 --eval_attacks
