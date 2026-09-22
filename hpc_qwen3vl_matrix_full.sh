#!/bin/bash
#SBATCH --job-name=qwen3vl_matrix_full
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/qwen3vl_matrix_full_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/qwen3vl_matrix_full_%j.err

# ============================================================
# DO NOT SUBMIT THIS until hpc_qwen3vl_diagnostic.sh has actually
# succeeded and shown Qwen3-VL clearing (or at least engaging
# differently than) LLaVA-NeXT's 2-turn ceiling. This is the full
# 629-task matrix -- same ~24-30 hour ballpark as the completed
# LLaVA-NeXT run -- committed to an unvalidated model wastes exactly
# the scarce biggpu time the diagnostic exists to protect.
#
# --output_dir is DELIBERATELY separate from agentdojo_matrix_full
# (LLaVA-NeXT's data) -- never share an output_dir between two
# different models; the resume/dedup logic keys purely on sample_id,
# which says nothing about which model produced it.
#
# Unlike the LLaVA-NeXT run, this starts with everything already
# enabled from the start (all three spotlighting defenses, both
# measurement flags) -- since this is Qwen3-VL's first run, there's
# nothing to retrofit or backfill later the way the LLaVA-NeXT
# backfill_measurements.py script had to.
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
export WANDB_MODE=offline

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

python experiments/run_attack_matrix.py \
    --agentdojo \
    --hpc \
    --model qwen3vl \
    --attack_names template_ipi gcg \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding \
    --enable_task_tracker \
    --enable_circuit_breaker_measurement \
    --use_wandb --wandb_project ipi-latent-analysis --wandb_run_name qwen3vl_full_biggpu \
    --output_dir qwen3vl_matrix_full

echo "Job finished (or was killed by walltime -- check squeue/sacct) at $(date)"
echo "If this was a walltime kill, just resubmit this exact script -- it resumes automatically."
