#!/bin/bash
#SBATCH --job-name=asr_reach
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/asr_reach_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/asr_reach_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python /tmp/asr_reach.py
