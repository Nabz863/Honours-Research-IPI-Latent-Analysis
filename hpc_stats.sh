#!/bin/bash
#SBATCH --job-name=stats
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/stats_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/stats_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python analysis/stats.py --results_dir vwa_adv_qwen3vl_pixel --layer 31 --n_perm 500 --out analysis_out/stats_vwa_pixel.json
