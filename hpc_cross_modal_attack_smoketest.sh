#!/bin/bash
#SBATCH --job-name=cross_modal_smoketest
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster46,mscluster47,mscluster51,mscluster63,mscluster64,mscluster70,mscluster44,mscluster69,mscluster74,mscluster75,mscluster76
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/cross_modal_smoketest_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/cross_modal_smoketest_%j.err

set -uo pipefail
mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
python3 experiments/smoketest_cross_modal_attack.py --iters 8
echo "Finished at $(date)"
