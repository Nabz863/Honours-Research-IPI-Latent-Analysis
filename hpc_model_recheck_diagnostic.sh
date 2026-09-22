#!/bin/bash
#SBATCH --job-name=model_recheck_diagnostic
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/model_recheck_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/model_recheck_%j.err

# Properly submitted via sbatch this time -- a bare `python3` command run
# directly in an interactive shell landed on whatever tiny GPU (7.6GB)
# was visible there, nowhere near enough for a 7B model, and violates
# the cluster's own "use sbatch or srun" guideline. Also runs a real
# file inside the repo, not a /tmp script -- /tmp is very likely
# node-local, so a file created on the login node wouldn't necessarily
# exist on whatever physical node this job lands on.
#
# Which model this checks is controlled by MODEL_TO_CHECK below --
# change it and resubmit rather than needing a second copy of this
# script. Defaults to llava, since that's the immediate real question
# (did the query()/_parse_model_output fixes change LLaVA-NeXT's
# behavior on the same 5 tasks it originally succeeded on).

set -euo pipefail

MODEL_TO_CHECK="${MODEL_TO_CHECK:-llava}"

mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate

export HF_HOME=/datasets/nmahomed/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Checking model: $MODEL_TO_CHECK"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

python experiments/qwen3vl_diagnostic.py --model "$MODEL_TO_CHECK"

echo "Job finished at $(date)"
