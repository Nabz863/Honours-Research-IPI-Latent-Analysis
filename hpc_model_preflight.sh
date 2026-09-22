#!/bin/bash
#SBATCH --job-name=model_preflight
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=mscluster76,mscluster47,mscluster64,mscluster70,mscluster74,mscluster51,mscluster63
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/model_preflight_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/model_preflight_%j.err

# Deliberately targets bigbatch (48 nodes), NOT biggpu (4 nodes, currently
# occupied by the three 71hr v2 matrix jobs) -- purely for queue speed,
# so this comes back in minutes rather than waiting days. This is NOT a
# validation run -- it only answers what the sandbox tests genuinely
# couldn't: does the real checkpoint actually download and load, does
# target_layers resolve to something sane on the REAL (not tiny) model,
# does ONE real clean episode run through the full agentic pipeline
# without crashing.
#
# Weights (~16-17GB bf16) are close to bigbatch's 24GB ceiling -- tighter
# than LLaVA-NeXT's 14GB had. If this itself OOMs even on a short
# episode, that's real, useful information learned in minutes: this
# model needs biggpu specifically, not a bug to fix.

MODEL="${MODEL:-internvl}"

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
echo "Pre-flight checking model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

python experiments/run_attack_matrix.py \
    --agentdojo \
    --hpc \
    --model "$MODEL" \
    --suites workspace \
    --max_tasks 2 \
    --attack_names template_ipi \
    --output_dir "model_preflight_${MODEL}"

echo "Job finished at $(date)"
echo "If this succeeded, $MODEL is confirmed working -- confident for its v2 matrix turn."
echo "If it OOM'd, $MODEL needs biggpu specifically -- worth knowing before its queue turn, not after."
