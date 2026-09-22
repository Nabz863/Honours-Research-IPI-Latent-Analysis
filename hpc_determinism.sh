#!/bin/bash
#SBATCH --job-name=determinism
#SBATCH --partition=biggpu
#SBATCH --nodes=1 --cpus-per-task=8 --mem=48G --time=00:30:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/determinism_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/determinism_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache TRANSFORMERS_CACHE=$HF_HOME HF_HUB_OFFLINE=1
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python experiments/determinism_check.py --model qwen3vl
