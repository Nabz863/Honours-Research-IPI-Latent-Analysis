#!/bin/bash
#SBATCH --job-name=determinism_check
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/determinism_check_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/determinism_check_%j.err

# Answers one question before anything else: is .generate() deterministic?
# query() calls it with no do_sample/temperature override, so whatever
# each model's own generation_config.json defaults to is what's actually
# running. This determines whether task_14's behavior change (correct
# email search pre-fix -> hallucinated calendar event post-fix) reflects
# something the parsing fix genuinely revealed, or is just sampling
# noise -- which would mean EVERY single-sample comparison collected in
# this project so far needs to be treated as one noisy draw, not fact.

set -euo pipefail

MODEL_TO_CHECK="${MODEL_TO_CHECK:-llava}"
TASK_TO_CHECK="${TASK_TO_CHECK:-user_task_14}"

mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate

export HF_HOME=/datasets/nmahomed/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Checking determinism: model=$MODEL_TO_CHECK task=$TASK_TO_CHECK"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

python experiments/determinism_check.py --model "$MODEL_TO_CHECK" --user_task_id "$TASK_TO_CHECK"

echo "Job finished at $(date)"
