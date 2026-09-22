#!/bin/bash
#SBATCH --job-name=coverage
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=04:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/coverage_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/coverage_%j.err
set -uo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python analysis/coverage_audit.py --out analysis_out/coverage.json
