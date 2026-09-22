#!/bin/bash
#SBATCH --job-name=agentdojo_matrix_full
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster76,mscluster47,mscluster64,mscluster70,mscluster74,mscluster51
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_full_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/agentdojo_matrix_full_%j.err

# ============================================================
# GPU NOTE: job 23523 landed on a node (mscluster76) with no visible GPU
# (nvidia-smi itself reported "No devices were found"). --gres=gpu:1 was
# tried as a fix and was WRONG -- `sinfo -p bigbatch -o "%N %G %t"`
# confirmed GRES=(null) on every single node in this partition, and
# sbatch rejected the flag outright ("Invalid generic resource (gres)
# specification"). This cluster does not track GPUs through SLURM's
# resource system at all -- getting one is purely a function of which
# physical node you land on, with no SLURM-level guarantee or isolation.
# That also means SLURM's own scheduler has no way to route around a
# node with a dead/hung GPU driver by itself.
#
# Current mitigation, in order of what actually helps:
#   1. --exclude below rules out the one node (76) already confirmed bad,
#      plus the ones sinfo showed as down/drain (SLURM wouldn't schedule
#      to those anyway, but excluding explicitly costs nothing).
#   2. experiments/run_attack_matrix.py's _require_gpu_if_hpc() fails
#      within SECONDS if a job lands on another bad node, instead of
#      silently grinding on CPU for hours before anyone notices -- this
#      is now the real safety net, since SLURM itself provides none here.
#   3. If bad-node landings keep recurring beyond mscluster76, this is a
#      cluster infrastructure problem, not something a script can fully
#      route around -- worth raising with Shunmuga directly (GRES=(null)
#      across the whole partition, post-rebuild, is itself worth a
#      report even independent of this specific failure).
#
# Real, full-scale AgentDojo run: clean + 2 attacks (template_ipi, gcg)
# + 3 spotlighting variants, each applied to clean AND every attack --
# 1 + 2 + 3 + (2*3) = 12 conditions per task, across ALL FOUR suites
# (no --suites given = all of them).
#
# Rough budget from real per-task numbers seen so far (clean/template_ipi
# ~13s, gcg ~53s, defended episodes ~similar to their undefended
# counterpart): for the full ~629-task matrix this is in the region of
# 1.5-2 days. --time=48:00:00 is a guess at this cluster's max walltime,
# NOT a confirmed value -- check with Shunmuga or `sacctmgr show qos` if
# you want a real number instead of this guess.
#
# THIS IS SAFE TO JUST RESUBMIT if SLURM kills it on walltime: the
# updated run_attack_matrix.py checkpoints results_log,
# adv_payloads_cache.json, and clean_activations_cache.pt after EVERY
# single episode, and skips any sample_id already present on startup --
# including reusing already-computed (expensive) gcg payloads rather
# than regenerating them. Verified in the sandbox with a genuine
# partial-run-then-resume test before this script was written. Just
# `sbatch hpc_agentdojo_matrix_full.sh` again with the exact same
# --output_dir if it gets killed -- no flags need to change.
#
# obfuscated_activation is deliberately NOT included in this first full
# run -- it's real-GPU-confirmed working and reasonably fast now (fixed,
# ~49s/task), but adding a third attack (and therefore a third
# defended-attack combination per defense) to an already ~2-day estimate
# compounds the walltime-limit risk on an ALREADY-long single job. Add
# it once this run's real total time is known, in a follow-up job (the
# clean pass and its cached activations will be reused, not redone,
# since it's the same --output_dir).
#
# WANDB_MODE=offline: compute-node internet access on this cluster is
# unconfirmed -- login-node `wandb login` succeeding does NOT establish
# that bigbatch nodes can reach api.wandb.ai. Sync from the login node
# after (or during, periodically) via: wandb sync <path-to-run-dir>
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
    --use_wandb --wandb_project ipi-latent-analysis --wandb_run_name agentdojo_full_llava \
    --output_dir agentdojo_matrix_full

echo "Job finished (or was killed by walltime -- check squeue/sacct) at $(date)"
echo "If this was a walltime kill, just resubmit this exact script -- it resumes automatically."
