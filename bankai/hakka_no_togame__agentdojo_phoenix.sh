#!/bin/bash
# hakka_no_togame__agentdojo_phoenix -- AgentDojo v1.1.2 for the extension models
# (LLaVA-NeXT, InternVL3) on PRIME's phoenix: clean, attacks, defences.
# ---------------------------------------------------------------------------
# Same attacks, defences and flags as the core models' runs (tensa_zangetsu /
# senbonzakura_kageyoshi). Different hardware from the core models: phoenix's
# two RTX 4090s (24 GB each, native bf16), BOTH used by one model (layers split
# across the two cards; a single 24 GB card ran out of memory on workspace and
# travel). One model at a time.
# Resumable: prime's walltime is 3 days; if a run hits it, resubmit the SAME
# command -- completed rows are skipped, errored rows retried. gpu_name is in
# run_settings, so a resume is refused on any other GPU type: keep -w phoenix.
# Probe 66500 (LLaVA, 12 pairs): ~27-31 s per episode, peak 22 GB
#   => ~70 h for the full matrix, so expect one resubmit per model.
#   sbatch --export=ALL,MODEL=llava    bankai/hakka_no_togame__agentdojo_phoenix.sh
#   sbatch --export=ALL,MODEL=internvl bankai/hakka_no_togame__agentdojo_phoenix.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=hakka_no_togame__agentdojo_phoenix
#SBATCH --partition=prime
#SBATCH --nodelist=phoenix
#SBATCH --nodes=1 --cpus-per-task=8 --mem=96G --time=3-00:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/hakka_no_togame__agentdojo_phoenix_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/hakka_no_togame__agentdojo_phoenix_%j.err

set -uo pipefail
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
export HF_HOME=/datasets/nmahomed/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ICON_ORACLE_ATTACKED=1          # ICON = rectifier under oracle detection (decision 3 Oct)
export CUDA_VISIBLE_DEVICES=0,1        # both 4090s for one model
MODEL="${MODEL:?set MODEL=llava or MODEL=internvl}"
case "$MODEL" in llava|internvl) ;; *) echo "FATAL: MODEL must be llava or internvl" >&2; exit 1;; esac
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)   model: $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() == 2 else 1)" \
    || { echo "FATAL: expected 2 CUDA devices on $(hostname)" >&2; exit 1; }
export AGENTDOJO_BENCHMARK_VERSION=v1.1.2
OUT="$R/final_agentdojo_${MODEL}"
# --- code freeze guard: every row's code_version must show a clean commit ---
if git -C "$R" status --porcelain --untracked-files=all -- attacks defenses environments experiments extraction | grep -q '\.py$'; then
    echo "FATAL: uncommitted .py changes in the code directories -- commit first:" >&2
    git -C "$R" status --porcelain -- attacks defenses environments experiments extraction | grep '\.py$' >&2
    exit 1
fi
echo "code: $(git -C "$R" rev-parse --short HEAD)"

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
