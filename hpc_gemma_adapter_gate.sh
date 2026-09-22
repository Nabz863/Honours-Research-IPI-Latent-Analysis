#!/bin/bash
#SBATCH --job-name=gemma_gate
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=00:30:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/gemma_gate_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/gemma_gate_%j.err
set -uo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache TRANSFORMERS_CACHE=$HF_HOME HF_HUB_OFFLINE=1
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
IMG=/datasets/nmahomed/vwa-adv/agent-attack/exp_data/agent_adv/reddit_142_wrong_text_cap/victim_image.png
python -m attacks.geometry_adapters --model_id google/gemma-4-12B-it --image "$IMG" --atol 2.5e-1
