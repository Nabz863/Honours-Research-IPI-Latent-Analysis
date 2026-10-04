#!/bin/bash
# senbonzakura_kageyoshi__agentdojo_gemma -- AgentDojo v1.1.2, gemma4: clean, attacks, defences
# ---------------------------------------------------------------------------
# Final run (P2a.1 code freeze). Resumable: if it hits the walltime, resubmit
# the SAME script; completed rows are skipped, errored rows retried.
# Blackwell nodes only (110/112): one GPU type per model, so a resume can never
# mix hardware (gpu_name is in run_settings and a mismatch is refused).
# Estimated: ~35-40 h (8,662 episodes at ~14 s; ICON slower)
# ---------------------------------------------------------------------------
#SBATCH --job-name=senbonzakura_kageyoshi__agentdojo_gemma
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster106,mscluster107,mscluster108,mscluster109,mscluster111
#SBATCH --nodes=1 --cpus-per-task=8 --mem=96G --time=3-00:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/senbonzakura_kageyoshi__agentdojo_gemma_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/senbonzakura_kageyoshi__agentdojo_gemma_%j.err

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
export AGENTDOJO_BENCHMARK_VERSION=v1.1.2
MODEL=gemma4
OUT="$R/final_agentdojo_${MODEL}"
# --- code freeze guard: every row's code_version must show a clean commit ---
if git -C "$R" status --porcelain --untracked-files=all -- attacks defenses environments experiments extraction | grep -q '\.py$'; then
    echo "FATAL: uncommitted .py changes in the code directories -- commit first:" >&2
    git -C "$R" status --porcelain -- attacks defenses environments experiments extraction | grep '\.py$' >&2
    exit 1
fi
echo "code: $(git -C "$R" rev-parse --short HEAD)"

# Order = priority: clean, the two text attacks, then the defences in the order
# listed (spotlighting first, Attention Tracker last), so a cut falls on the end.
# GCG / obfuscated activation (P2c) are appended later into the SAME directory.
python experiments/run_attack_matrix.py --agentdojo --model "$MODEL" --hpc \
    --attack_names template_ipi important_instructions \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding icon attention_tracker \
    --dedup_defended_clean \
    --output_dir "$OUT"
echo "=== run exited $? at $(date) ==="

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
