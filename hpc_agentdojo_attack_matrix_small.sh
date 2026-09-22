#!/bin/bash
#SBATCH --job-name=agentdojo_matrix_small
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_small_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_small_%j.err

# ============================================================
# First REAL run of run_agentdojo_attack_matrix() -- one suite
# (workspace), 10 tasks, template_ipi only. Deliberately small: this is
# about confirming the real 7B model + real multi-task loop + real
# per-episode activation saving all hold up together before committing
# GPU time to the full ~629-task matrix across all 4 suites.
#
# Same OOM mitigations as the smoketest (job 22415, which completed
# clean): empty_cache() per turn already in agentdojo_llm.py,
# expandable_segments here, and max_new_tokens capped in the pipeline
# element's default.
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

python experiments/run_attack_matrix.py \
    --agentdojo \
    --hpc \
    --suites workspace \
    --max_tasks 10 \
    --attack_names template_ipi \
    --output_dir agentdojo_matrix_workspace_small

echo "Job finished at $(date)"
