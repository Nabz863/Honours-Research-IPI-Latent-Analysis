#!/bin/bash
#SBATCH --job-name=convert_postmill
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/convert_postmill_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/convert_postmill_%j.err

# ============================================================
# One-time Apptainer conversion: postmill.tar (Docker export) -> postmill.sif
# Wits HPC (SLURM) job script -- mscluster bigbatch partition
#
# Why sbatch, not local/login node: the tar is ~53GB with a huge
# node_modules tree, which is what ran out of RAM running it locally.
# This is CPU/IO-bound, not GPU-bound -- it just needs a compute node with
# real RAM and scratch space, which bigbatch provides even as a GPU
# partition (no GPU is requested or needed here).
#
# Previous attempt (job 1013) did NOT fail on a real error -- every
# "ignoring (usually) harmless EPERM on setxattr" line is Apptainer's
# normal rootless-build noise. It was killed "DUE TO TIME LIMIT" after
# ~6.5 minutes, still extracting node_modules/.cache/terser-webpack-plugin
# (tens of thousands of tiny cache files). The fix is overwhelmingly a
# longer --time, not anything structural -- mem is bumped up too since
# that's the other thing that's bitten this conversion before.
#
# Submit with: sbatch hpc_convert_postmill.sh
# ============================================================

set -euo pipefail

mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

# ---- Apptainer scratch/cache: /datasets has room, home dir doesn't -----
# (same class of mistake as weights landing in $HOME instead of /datasets
# -- keep every large scratch/cache dir pointed at /datasets explicitly)
export APPTAINER_TMPDIR=/datasets/nmahomed/apptainer_tmp
export APPTAINER_CACHEDIR=/datasets/nmahomed/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ---- Paths -- adjust TAR_PATH if the transferred file has a different name
TAR_PATH=/datasets/nmahomed/containers/postmill-populated-exposed-withimg.tar
SIF_OUTPUT=/datasets/nmahomed/containers/sif/postmill.sif

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Apptainer version: $(apptainer --version)"
echo "Tar: $(du -h "$TAR_PATH" 2>/dev/null || echo "NOT FOUND at $TAR_PATH -- check TAR_PATH")"
echo "Free space on /datasets: $(df -h /datasets | tail -1)"

# ---- Build --------------------------------------------------------------
apptainer build --force "$SIF_OUTPUT" docker-archive://"$TAR_PATH"

echo "Job finished at $(date)"
echo "Output: $(ls -lh "$SIF_OUTPUT" 2>/dev/null || echo 'SIF NOT FOUND -- build did not complete')"
