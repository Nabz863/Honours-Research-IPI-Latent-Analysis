#!/bin/bash
#SBATCH --job-name=vwa_adv_episode_smoketest
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/vwa_adv_episode_smoketest_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/vwa_adv_episode_smoketest_%j.err

# No --exclude here on purpose: --nodelist at submission time already
# pins this to one specific node (wherever Postmill landed), which by
# construction isn't one of the excluded nodes since serve_postmill.slurm
# excludes them too. Submit with:
#
#   sbatch --nodelist=<postmill_hostname> hpc_vwa_adv_episode_smoketest.sh <postmill_hostname>
#
# <postmill_hostname> is printed by serve_postmill.slurm's own .out log
# once it's up -- look for the "NODE FOR THE EPISODE JOB" banner.

set -uo pipefail
mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

POSTMILL_HOST="${1:?Usage: sbatch --nodelist=<host> hpc_vwa_adv_episode_smoketest.sh <host>}"
export REDDIT="http://${POSTMILL_HOST}:8080"
export VWA_ADV_REPO=/datasets/nmahomed/vwa-adv/visualwebarena

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Expecting Postmill at $REDDIT"

if [ "$(hostname)" != "$POSTMILL_HOST" ]; then
    echo "WARNING: this node ($(hostname)) does not match the Postmill host you passed"
    echo "($POSTMILL_HOST) -- --nodelist likely wasn't set at submission, or wasn't"
    echo "honored. Cross-node access to Postmill's Apptainer instance is NOT expected"
    echo "to work (see the same-node constraint noted in serve_postmill.slurm)."
fi

python3 experiments/run_one_vwa_adv_episode.py --task-id 0
echo "Finished at $(date)"
