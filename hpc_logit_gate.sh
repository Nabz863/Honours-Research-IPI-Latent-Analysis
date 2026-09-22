#!/bin/bash
#SBATCH --job-name=logit_gate
#SBATCH --partition=biggpu
#SBATCH --nodes=1 --cpus-per-task=8 --mem=48G --time=01:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/logit_gate_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/logit_gate_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache TRANSFORMERS_CACHE=$HF_HOME HF_HUB_OFFLINE=1
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
G='/datasets/nmahomed/vwa-adv/agent-attack/exp_data/agent_adv/reddit_*/victim_image.png'
for M in gemma4; do
  echo "############ $M"
  python experiments/adapter_logit_gate.py --model $M --image_glob "$G" \
      --n_images 5 --out analysis_out/logit_gate_$M.json
done
