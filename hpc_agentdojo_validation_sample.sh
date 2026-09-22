#!/bin/bash
#SBATCH --job-name=agentdojo_validation_sample
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_validation_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_validation_%j.err

# Medium-scale validation of the FIXED pipeline (parser fix + confirmed
# deterministic generation) BEFORE committing to a full 629-task re-run.
# Specifically measures: (1) how often completions look sane vs
# hallucinated/degenerate under the fix, (2) what fraction of episodes
# hit ToolsExecutionLoop's real max_iters=15 ceiling -- task_17 in the
# 5-task check did, taking ~6min instead of ~20s, and how COMMON that
# is directly determines whether the full re-run stays near its
# original ~24-30hr estimate or runs substantially longer.
#
# 15 tasks/suite x 4 suites = ~60 tasks, clean + template_ipi only (no
# GCG, no defenses) -- cheapest, fastest signal on prevalence and
# timing, not a second full matrix. Separate --output_dir so this never
# touches or gets confused with agentdojo_matrix_full's (partially
# untrustworthy, pre-fix) data.

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
    --model llava \
    --suites workspace banking slack travel \
    --max_tasks_per_suite 15 \
    --attack_names template_ipi \
    --use_wandb --wandb_project ipi-latent-analysis --wandb_run_name agentdojo_validation_sample \
    --output_dir agentdojo_matrix_validation

echo "Job finished at $(date)"
