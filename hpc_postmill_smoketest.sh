#!/bin/bash
#SBATCH --job-name=postmill_smoketest
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/postmill_smoketest_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/postmill_smoketest_%j.err

# ============================================================
# Second attempt at running postmill.sif. The first attempt's
# `apptainer instance start` succeeded but launched nothing -- its own
# instance metadata showed the process was `/.singularity.d/actions/start`,
# which is Apptainer's %startscript. Docker images have no equivalent to
# that; ENTRYPOINT/CMD only ever maps to the %runscript, invoked by
# `apptainer run`, not `instance start`. So `instance start` alone just
# holds the container's namespaces open -- nothing gets executed inside
# them until `apptainer run instance://NAME` is called separately. That's
# what actually triggers Postmill's baked-in entrypoint (likely a
# supervisor managing apache + the bundled, pre-populated DB).
# ============================================================

set -uo pipefail

export APPTAINER_TMPDIR=/datasets/nmahomed/apptainer_tmp
export APPTAINER_CACHEDIR=/datasets/nmahomed/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
mkdir -p /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs

SIF=/datasets/nmahomed/containers/sif/postmill.sif
INSTANCE_NAME=forum
RUN_LOG=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/postmill_run_${SLURM_JOB_ID}.log

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

# Clean up any stale registration from a previous attempt before starting fresh.
apptainer instance stop "$INSTANCE_NAME" 2>/dev/null
sleep 2

apptainer instance start --writable-tmpfs "$SIF" "$INSTANCE_NAME"
echo "instance start exit code: $?"
apptainer instance list

# The actual fix: invoke the runscript (Docker's baked-in ENTRYPOINT/CMD)
# inside the now-running instance. Backgrounded since this is the
# foreground service process itself.
apptainer run instance://"$INSTANCE_NAME" > "$RUN_LOG" 2>&1 &
RUN_PID=$!
echo "apptainer run backgrounded as PID $RUN_PID, logging to $RUN_LOG"

# Poll rather than a fixed sleep -- a bundled DB server can take a while
# to finish its own startup sequence on first real boot.
echo "--- Polling for a response (up to 2 minutes) ---"
READY=0
for i in $(seq 1 24); do
    sleep 5
    for port in 80 9999 8080; do
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:${port}/" 2>/dev/null)
        if [ -n "$code" ] && [ "$code" != "000" ]; then
            echo "  [$(date +%T)] port $port -> HTTP $code"
            READY=1
        fi
    done
    if [ "$READY" -eq 1 ]; then
        break
    fi
    echo "  [$(date +%T)] still waiting... (${i}x5s elapsed)"
done

echo ""
echo "--- Listening ports now ---"
ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null

echo ""
echo "--- Run log ($RUN_LOG) ---"
cat "$RUN_LOG" 2>/dev/null || echo "(empty or not found)"

echo ""
echo "--- Final reachability check ---"
for port in 80 9999 8080; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:${port}/" 2>/dev/null)
    echo "  port $port -> HTTP ${code:-no response}"
done

echo ""
echo "Keeping the job (and the backgrounded apptainer run process) alive"
echo "briefly for inspection. Stop everything with: apptainer instance stop ${INSTANCE_NAME}"
echo "Job finished at $(date)"

sleep 300
