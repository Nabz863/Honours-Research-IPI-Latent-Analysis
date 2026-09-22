#!/bin/bash
#SBATCH --job-name=agentdojo_matrix_obf_small
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster76,mscluster47,mscluster64,mscluster70,mscluster74,mscluster51
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_obf_small_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_obf_small_%j.err

# ============================================================
# First real-GPU run of obfuscated_activation.py. GCG (job 23409) just
# proved that "verified against a fake model" does not mean "safe
# against real weights" -- the YAML crash (job 23406) was invisible
# until real weights hit real AgentDojo. obfuscated_activation shares
# GCG's core mechanics (same fix already applied to both) but has its
# own independent candidate-selection/seed-validation code path and
# additional per-candidate extractor hook overhead GCG doesn't have --
# worth confirming it actually runs clean on real hardware before
# folding it into a bigger combined job.
#
# template_ipi/gcg deliberately NOT re-run here -- both already have
# real, confirmed numbers (job 23409). Only testing what's still unknown.
#
# --time=01:00:00 for 3 tasks is generous given job 23409's real timing
# (gcg's ~53s/task suggests obfuscated_activation, which does similar
# work plus hook overhead, shouldn't be wildly more expensive -- but
# there's no real data yet, hence the padding).
# ============================================================

set -euo pipefail

mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate

export HF_HOME=/datasets/nmahomed/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

time python experiments/run_attack_matrix.py \
    --agentdojo \
    --hpc \
    --suites workspace \
    --max_tasks 3 \
    --attack_names obfuscated_activation \
    --output_dir agentdojo_matrix_workspace_obf_small

echo "Job finished at $(date)"
