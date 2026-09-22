#!/bin/bash
#SBATCH --job-name=agentdojo_matrix_full
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_full_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_full_%j.err

# ============================================================
# Moved from bigbatch (RTX 3090, 24GB) to biggpu (Quadro RTX 8000,
# 48GB) per the real mscluster Community Guidelines: bigbatch is
# uniformly 24GB, NOT a mix of 3090/8000 as earlier notes assumed --
# every job this whole project has run on bigbatch confirms exactly
# that. ~45% of workspace-suite clean episodes were OOMing at ~21-22GB
# used against bigbatch's 24GB ceiling (jobs 23527/23613/23687) --
# biggpu's 48GB per card removes that ceiling as a concern entirely,
# with no precision change and no code change, unlike int8/context-
# truncation alternatives.
#
# biggpu is explicitly scarce (4 nodes total, "very expensive to run
# and maintain... in high demand") and reserved by policy for mature,
# debugged code -- this qualifies given how extensively tested and
# OOM-resilient this pipeline now is, but it's still someone else's
# scarce resource: --cpus-per-task/--mem deliberately left modest
# (matching what bigbatch used, not scaled up to grab everything
# biggpu's 56 cores/1TB RAM could offer) since none of that is the
# actual bottleneck here, only GPU VRAM is. Only requesting 1 of the
# 2 GPUs on the node for the same reason -- don't take both if the
# job only needs one.
#
# Resume/OOM-safety/W&B fixes all carry over unchanged from the
# bigbatch version -- this is the SAME --output_dir
# (agentdojo_matrix_full), so it picks up exactly where the last
# bigbatch run left off, no flags changed beyond --partition.
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
    --attack_names template_ipi gcg \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding \
    --use_wandb --wandb_project ipi-latent-analysis --wandb_run_name agentdojo_full_llava_biggpu \
    --output_dir agentdojo_matrix_full

echo "Job finished (or was killed by walltime -- check squeue/sacct) at $(date)"
echo "If this was a walltime kill, just resubmit this exact script -- it resumes automatically."
