#!/bin/bash
#SBATCH --job-name=gpu_healthcheck
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster44,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster70,mscluster69,mscluster74,mscluster75,mscluster76
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/gpu_healthcheck_%A_%a.out

source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
echo "Node: $(hostname)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
