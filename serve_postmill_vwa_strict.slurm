#!/bin/bash
#SBATCH --job-name=postmill_serve
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/postmill_serve_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/postmill_serve_%j.err

# Exclude list matches the known-bad GPU nodes from the AgentDojo work --
# added here too because whatever node Postmill lands on is also where
# the episode job (needs a real GPU for LLaVA-NeXT) will be pinned via
# --nodelist. A GPU-broken node breaks that pairing even though Postmill
# itself doesn't need a GPU.

set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs
mkdir -p /var/tmp/nmahomed_apptainer_run
export APPTAINER_TMPDIR=/var/tmp/nmahomed_apptainer_run
export APPTAINER_CACHEDIR=/datasets/nmahomed/apptainer_cache

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

# bigbatch has GRES=(null) -- SLURM does not track GPUs, so it will happily
# schedule onto a node whose GPU is broken or absent. The --exclude list only
# grows by discovery and is therefore always incomplete. Verify directly and
# fail in seconds, rather than after Postmill is up and a 7B model has been
# loaded onto the CPU at ~100x slowdown (which is exactly what happened on
# mscluster46 -- it served Postmill fine, since that needs no GPU, and the
# problem only surfaced much later).
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "FATAL: no working CUDA device on $(hostname). Add it to --exclude and resubmit:"
    echo "  sed -i '4s/$/,$(hostname -s)/' serve_postmill.slurm && sbatch serve_postmill.slurm"
    exit 1
fi
echo "GPU check passed on $(hostname)."


apptainer instance stop postmill_instance 2>/dev/null || true
sleep 2

apptainer instance start --fakeroot --no-home --writable-tmpfs \
    /datasets/nmahomed/containers/sif/postmill.sif \
    postmill_instance
sleep 5

echo "Fixing permissions (marker is baked into the image -- always needed on a fresh instance)..."
apptainer exec --fakeroot --no-home instance://postmill_instance rm -f /run/postgresql/.init
time apptainer exec --fakeroot --no-home instance://postmill_instance \
    chown -R postgres:postgres /usr/local/pgsql/data /run/postgresql

echo "Patching nginx to a non-privileged port (also baked-in, also always needed)..."
apptainer exec --fakeroot --no-home instance://postmill_instance \
    sh -c "sed -i 's/listen 80;/listen 8080;/; s/listen \[::\]:80/listen [::]:8080/' /etc/nginx/conf.d/*.conf"

ENTRYPOINT_LOG=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/postmill_entrypoint_${SLURM_JOB_ID}.log

echo "Starting the real stack (nginx + postgres + php-fpm via supervisord)..."
apptainer run --fakeroot --no-home instance://postmill_instance \
    > "$ENTRYPOINT_LOG" 2>&1 &

# Everything above is real setup -- fail fast under set -e if any of it
# breaks. Everything below is a health check on something that legitimately
# takes an unknown amount of time to come up (cold postgres init); it must
# NOT be able to kill the backgrounded server process via a failed curl.
set +e

echo "=== health check ==="
READY=0
for i in $(seq 1 24); do   # up to 2 minutes
    sleep 5
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://localhost:8080/ 2>/dev/null)
    if [ -n "$code" ] && [ "$code" != "000" ]; then
        echo "  [$(date +%T)] HTTP $code -- Postmill is responding"
        READY=1
        break
    fi
    echo "  [$(date +%T)] not ready yet (${i}x5s elapsed)..."
done

if [ "$READY" -eq 1 ]; then
    echo "Postmill is up."
else
    echo "WARNING: no HTTP response after 2 minutes -- check $ENTRYPOINT_LOG"
    echo "Server process is still backgrounded regardless; this job keeps holding"
    echo "it up for the full walltime, so this is a warning, not a failed job."
fi

echo ""
echo "=========================================================="
echo "  NODE FOR THE EPISODE JOB: $(hostname)"
echo "  Submit hpc_vwa_adv_episode_smoketest.sh with:"
echo "    sbatch --nodelist=$(hostname) hpc_vwa_adv_episode_smoketest.sh $(hostname)"
echo "=========================================================="
echo ""
echo "Cancel with: scancel $SLURM_JOB_ID   (or it holds until the 8h walltime expires)"

R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"
export STRICT_ASR_CONFIG_DIR=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/strict_configs
source "$R/venv/bin/activate"
source ~/vwa_adv_env.sh
export AGENT_ATTACK_REPO=/datasets/nmahomed/vwa-adv/agent-attack
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== starting curated matrix at $(date) ==="
python experiments/run_vwa_adv_curated_matrix.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --attack_names template_ipi typographic steganographic adversarial_patch \
                   pixel_perturbation cross_modal_injection comet gcg \
    --site reddit --max_steps 15 \
    --output_dir "$R/vwa_adv_qwen3vl_strict"
echo "=== matrix finished at $(date) ==="
