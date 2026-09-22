#!/bin/bash
#SBATCH --job-name=vwa_adv_full
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster47,mscluster51,mscluster63,mscluster64,mscluster70,mscluster74,mscluster76
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/vwa_adv_full_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/vwa_adv_full_%j.err

set -uo pipefail
mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINER_TMPDIR=/var/tmp/nmahomed_apptainer_run
mkdir -p "$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR=/datasets/nmahomed/apptainer_cache
export REDDIT="http://localhost:8080"
export VWA_ADV_REPO=/datasets/nmahomed/vwa-adv/visualwebarena

cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

apptainer instance stop postmill_instance 2>/dev/null || true
sleep 2
apptainer instance start --fakeroot --no-home --writable-tmpfs \
    /datasets/nmahomed/containers/sif/postmill.sif postmill_instance
sleep 5

apptainer exec --fakeroot --no-home instance://postmill_instance rm -f /run/postgresql/.init
apptainer exec --fakeroot --no-home instance://postmill_instance \
    chown -R postgres:postgres /usr/local/pgsql/data /run/postgresql
apptainer exec --fakeroot --no-home instance://postmill_instance \
    sh -c "sed -i 's/listen 80;/listen 8080;/; s/listen \[::\]:80/listen [::]:8080/' /etc/nginx/conf.d/*.conf"

ENTRYPOINT_LOG=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/postmill_entrypoint_${SLURM_JOB_ID}.log
apptainer run --fakeroot --no-home instance://postmill_instance > "$ENTRYPOINT_LOG" 2>&1 &

echo "=== Postmill health check ==="
READY=0
for i in $(seq 1 24); do
    sleep 5
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://localhost:8080/ 2>/dev/null)
    if [ -n "$code" ] && [ "$code" != "000" ]; then
        echo "  [$(date +%T)] HTTP $code -- Postmill is responding"
        READY=1
        break
    fi
done

if [ "$READY" -eq 1 ]; then
    echo "=== Running one real VWA-Adv episode ==="
    python3 experiments/run_one_vwa_adv_episode.py --task-id 0
fi

echo "Job $SLURM_JOB_ID stays open on $(hostname). Attach: srun --jobid=$SLURM_JOB_ID --pty bash"
wait
