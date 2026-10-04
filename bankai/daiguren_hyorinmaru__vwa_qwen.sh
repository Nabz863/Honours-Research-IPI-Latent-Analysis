#!/bin/bash
# daiguren_hyorinmaru__vwa_qwen -- VWA-Adv (curated, reddit), Qwen/Qwen3-VL-8B-Instruct: clean, attacks, defences
# ---------------------------------------------------------------------------
# Final run (P2a.1 code freeze). Resumable: if it hits the walltime, resubmit
# the SAME script; completed rows are skipped, errored rows retried.
# Blackwell nodes only (110/112): one GPU type per model, so a resume can never
# mix hardware (gpu_name is in run_settings and a mismatch is refused).
# Estimated: ~25-35 h (1,920 episodes; ECSO ~3x)
# Runs through the dual-judge launcher, ALWAYS (the launcher is part of the
# harness hash, so every resume of this directory must use it too).
# ---------------------------------------------------------------------------
#SBATCH --job-name=daiguren_hyorinmaru__vwa_qwen
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster106,mscluster107,mscluster108,mscluster109,mscluster111
#SBATCH --nodes=1 --cpus-per-task=8 --mem=96G --time=3-00:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/daiguren_hyorinmaru__vwa_qwen_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/daiguren_hyorinmaru__vwa_qwen_%j.err

set -uo pipefail
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
export HF_HOME=/datasets/nmahomed/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ICON_ORACLE_ATTACKED=1          # ICON = rectifier under oracle detection (decision 3 Oct)
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "FATAL: no CUDA on $(hostname)" >&2; exit 1; }
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
OUT="$R/final_vwa_qwen3vl"
mkdir -p "$R/logs" "$OUT"
mkdir -p /var/tmp/nmahomed_apptainer_run
export APPTAINER_TMPDIR=/var/tmp/nmahomed_apptainer_run
export APPTAINER_CACHEDIR=/datasets/nmahomed/apptainer_cache
# --- code freeze guard: every row's code_version must show a clean commit ---
if git -C "$R" status --porcelain --untracked-files=all -- attacks defenses environments experiments extraction | grep -q '\.py$'; then
    echo "FATAL: uncommitted .py changes in the code directories -- commit first:" >&2
    git -C "$R" status --porcelain -- attacks defenses environments experiments extraction | grep '\.py$' >&2
    exit 1
fi
echo "code: $(git -C "$R" rev-parse --short HEAD)"

[ -s ~/.openai_key ] || { echo "FATAL: ~/.openai_key missing (judged tasks)" >&2; exit 1; }
JUDGE="meta-llama/Llama-3.1-8B-Instruct"
HF_HUB_OFFLINE=1 python3 -c "import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])" "$JUDGE" \
    || { echo "FATAL: judge weights not cached under $HF_HOME" >&2; exit 1; }

# --- Postmill (same preamble as the known-good VWA jobs) -----------------
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
apptainer run --fakeroot --no-home instance://postmill_instance \
    > "$R/logs/postmill_entrypoint_${SLURM_JOB_ID}.log" 2>&1 &
set +e
READY=0
for i in $(seq 1 24); do
    sleep 5
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://localhost:8080/ 2>/dev/null)
    if [ -n "$code" ] && [ "$code" != "000" ]; then READY=1; break; fi
done
[ "$READY" -eq 1 ] || { echo "FATAL: Postmill not responding" >&2; exit 1; }
echo "Postmill up."

export STRICT_ASR_CONFIG_DIR="$R/strict_configs"
export CLEAN_UA_CONFIG_DIR="$R/clean_configs"
source ~/vwa_adv_env.sh
export AGENT_ATTACK_REPO=/datasets/nmahomed/vwa-adv/agent-attack
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datasets/nmahomed/hf_cache
export ICON_ORACLE_ATTACKED=1
# The two LLM-judged tasks need the real key (commercial judge); never on the command line.
export OPENAI_API_KEY="$(cat ~/.openai_key)"


export HF_HOME=/datasets/nmahomed/hf_cache       # reassert after vwa_adv_env.sh

# Attention Tracker is NOT run on VWA-Adv (decision 4 Oct: every long page
# observation scores below the threshold, clean or attacked).
# P2b attacks (COMET, CrossInject, adversarial patch, pixel perturbation) and
# P2c (GCG, obfuscated activation) are appended later into the SAME directory.
python experiments/run_with_dual_judge.py \
    --dj_records "$OUT/judge_verdicts.jsonl" --dj_require_openai \
    --model_id "$MODEL_ID" \
    --attack_names template_ipi typographic steganographic bim_caption_transfer clip_attack_transfer \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding gaussian_blur_1.5 gaussian_blur_3.0 ecso icon \
    --site reddit --max_steps 15 \
    --output_dir "$OUT"
echo "=== run exited $? at $(date) ==="
apptainer instance stop postmill_instance 2>/dev/null || true

python3 - "$OUT" <<'PY'
import json, sys, collections
from pathlib import Path
rows = json.load(open(Path(sys.argv[1]) / "attack_results.json"))
c = collections.Counter((r.get("condition"), r.get("attack_name"), r.get("defense_name", "none")) for r in rows)
err = [r for r in rows if r.get("error")]
print(f"{len(rows)} rows, {len(err)} with an error (a resubmit retries them)")
for k in sorted(c, key=str):
    rs = [r for r in rows if (r.get("condition"), r.get("attack_name"), r.get("defense_name", "none")) == k and not r.get("error")]
    asr = sum(1 for r in rs if r.get("asr_success")); ua = sum(1 for r in rs if r.get("utility_success"))
    print(f"  {str(k):75s} n={c[k]:4d} ok={len(rs):4d} ASR={asr} UA={ua}")
PY
