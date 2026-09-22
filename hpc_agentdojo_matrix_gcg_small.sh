#!/bin/bash
#SBATCH --job-name=agentdojo_matrix_gcg_small
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster76,mscluster47,mscluster64,mscluster70,mscluster74,mscluster51
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_gcg_small_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_gcg_small_%j.err

# ============================================================
# First real-GPU run of gcg.py. Job 23405 only exercised template_ipi
# (0/10 ASR against real LLaVA-NeXT 7B) -- that sbatch script predates
# gcg.py/obfuscated_activation.py, so nothing else has run for real yet.
#
# Deliberately smaller than job 23405 (3 tasks, not 10) and deliberately
# does NOT also include obfuscated_activation in this same job -- that
# attack is strictly more expensive per candidate (extractor hook
# register/enter/exit overhead on top of the forward pass GCG already
# does) and shares the same core mechanics, so it makes sense to
# confirm GCG's real per-task cost first rather than stack two untested
# real-GPU costs into one job and not know which one ate the time.
#
# --time=02:00:00 for 3 tasks is generously padded -- there's no real
# timing data yet for GCG on this hardware, so this errs toward wasting
# idle allocation time rather than getting killed near the finish line.
# template_ipi is included again too (cheap, near-free) as a
# determinism/consistency check against job 23405's already-known 0/10
# result on the same suite.
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
    --attack_names template_ipi gcg \
    --output_dir agentdojo_matrix_workspace_gcg_small

echo "Job finished at $(date)"
