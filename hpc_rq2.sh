#!/bin/bash
#SBATCH --job-name=rq2
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/rq2_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/rq2_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python analysis/rq2_geometry.py --results_dir agentdojo_matrix_v4_qwen3vl \
    --layer 31 --out analysis_out/rq2_qwen3vl_v2.json
