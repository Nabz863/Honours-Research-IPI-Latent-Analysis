#!/bin/bash
#SBATCH --job-name=agentdojo_matrix_v2
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster111,mscluster107
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=71:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_v2_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_v2_%j.err

# ATTACKS-FIRST PHASE: defenses deliberately NOT run yet -- matches the
# proposal's own "Baseline Anchoring" methodology (3.4.2): run the full
# attack matrix unprotected first, carry only the most potent attacks
# forward into the defense matrix. Defended passes are the expensive
# part (3 defended-clean + 6 defended-attacked per task on top of
# clean+attacked -- deferring them is a 4x episode-count cut).
# TaskTracker/CircuitBreaker measurement stays ON -- free, computed from
# episodes already running, not separate passes.
#
# obfuscated_activation added: built, real-GPU-tested, previously just
# never turned on. num_steps/candidates_per_step raised from its own
# class defaults (30/32) to 50/64, same reasoning as GCG's own
# strengthening. Does NOT use GCG's new multi-restart mechanism --
# ObfuscatedActivationAttack has its own generate(), doesn't inherit it.
#
# FRESH output directory per model, not resuming into the original
# agentdojo_matrix_full/ -- that would silently skip ~7300 rows that
# "succeeded" under the now-fixed prompt-stripping/parser bug.
#
# ~76hr/model was estimated BEFORE this attacks-only cut and BEFORE
# GCG/obfuscated_activation strengthening -- treat as a rough anchor
# until real timing exists under the current config. Exceeds this
# cluster's 72hr MaxTime regardless -- expect at least one resume-based
# resubmit; that's expected, the resume/dedup logic makes it safe.
#
#   sbatch hpc_agentdojo_matrix_full_v2.sh                                        # llava (default)
#   MODEL=internvl OUTPUT_DIR=agentdojo_matrix_internvl sbatch hpc_agentdojo_matrix_full_v2.sh
#   MODEL=qwen3vl   OUTPUT_DIR=agentdojo_matrix_qwen3vl  sbatch hpc_agentdojo_matrix_full_v2.sh
set -euo pipefail
MODEL="${MODEL:-llava}"
OUTPUT_DIR="${OUTPUT_DIR:-agentdojo_matrix_full_v2}"
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
echo "Model: $MODEL | Output dir: $OUTPUT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python experiments/run_attack_matrix.py \
    --agentdojo \
    --hpc \
    --model "$MODEL" \
    --attack_names template_ipi gcg obfuscated_activation \
    --obfuscation_num_steps 50 \
    --obfuscation_candidates_per_step 64 \
    --enable_task_tracker \
    --enable_circuit_breaker_measurement \
    --use_wandb --wandb_project ipi-latent-analysis --wandb_run_name "agentdojo_v2_${MODEL}_attacks_only" \
    --output_dir "$OUTPUT_DIR"
echo "Job finished (or was killed by walltime -- check squeue/sacct) at $(date)"
echo "If this was a walltime kill, resubmit with the SAME MODEL/OUTPUT_DIR -- it resumes automatically."
echo "Defenses (--defense_names) deliberately NOT run in this phase -- see header comment."
