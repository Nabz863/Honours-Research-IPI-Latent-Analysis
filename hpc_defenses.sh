#!/bin/bash
#SBATCH --job-name=agentdojo_defenses
#SBATCH --partition=biggpu
#SBATCH --nodes=1 --cpus-per-task=8 --mem=48G --time=71:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/defenses_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/defenses_%j.err
set -euo pipefail
MODEL="${MODEL:-qwen3vl}"
OUTPUT_DIR="${OUTPUT_DIR:-agentdojo_matrix_v4_qwen3vl}"
export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache TRANSFORMERS_CACHE=$HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python experiments/run_attack_matrix.py --agentdojo --hpc --model "$MODEL" \
    --attack_names template_ipi \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding \
    --enable_task_tracker --enable_circuit_breaker_measurement \
    --use_wandb --wandb_project ipi-latent-analysis --wandb_run_name "defenses_${MODEL}" \
    --output_dir "$OUTPUT_DIR"
