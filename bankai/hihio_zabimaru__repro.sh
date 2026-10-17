#!/bin/bash
# hihio_zabimaru__repro -- run-to-run reproducibility of the final AgentDojo run
# ---------------------------------------------------------------------------
# Re-runs a fixed sample of 50 episodes (10 user tasks x clean, both attacks,
# delimiting, ICON; analysis/repro_check.py select) into a FRESH directory with
# the same code and flags as the final run, on the same GPU type (Blackwell;
# 110 and 112 are down, so 111 -- the band re-runs already wrote into the final
# directories from 111, which the runner's gpu_name check would have refused
# had its GPU type differed),
# then compares them with the originals: scores, payloads, every prompt hash and
# completion, tool calls, and every saved activation. Read-only on the final run.
# Resumable (resubmit the same command). ~1 h per model.
#   sbatch --export=ALL,MODEL=gemma4  bankai/hihio_zabimaru__repro.sh
#   sbatch --export=ALL,MODEL=qwen3vl bankai/hihio_zabimaru__repro.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=hihio_zabimaru__repro
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster106,mscluster107,mscluster108,mscluster109,mscluster110,mscluster112
#SBATCH --nodes=1 --cpus-per-task=8 --mem=96G --time=06:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/hihio_zabimaru__repro_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/hihio_zabimaru__repro_%j.out

set -uo pipefail
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
export HF_HOME=/datasets/nmahomed/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ICON_ORACLE_ATTACKED=1          # as in the final run
export AGENTDOJO_BENCHMARK_VERSION=v1.1.2
MODEL="${MODEL:?set MODEL=gemma4 or MODEL=qwen3vl}"
case "$MODEL" in gemma4|qwen3vl) ;; *) echo "FATAL: MODEL must be gemma4 or qwen3vl" >&2; exit 1;; esac
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)   model: $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "FATAL: no CUDA on $(hostname)" >&2; exit 1; }
if git -C "$R" status --porcelain --untracked-files=all -- attacks defenses environments experiments extraction | grep -q '\.py$'; then
    echo "FATAL: uncommitted .py changes in the code directories -- commit first:" >&2
    git -C "$R" status --porcelain -- attacks defenses environments experiments extraction | grep '\.py$' >&2
    exit 1
fi
echo "code: $(git -C "$R" rev-parse --short HEAD)"
ORIG="$R/final_agentdojo_${MODEL}"
OUT="$R/repro_agentdojo_${MODEL}"
SAMPLES="$R/analysis_out/repro_samples_${MODEL}.txt"
mkdir -p "$R/analysis_out" "$OUT"
# Selected once: a resubmit keeps the same sample even if the final directory
# has changed since (P2c rows, retries).
if [ ! -s "$SAMPLES" ]; then
    python3 analysis/repro_check.py select "$ORIG" > "$SAMPLES.tmp" && mv "$SAMPLES.tmp" "$SAMPLES" || exit 1
fi
echo "sample: $(grep -c . "$SAMPLES") episodes -> $SAMPLES"

python experiments/run_attack_matrix.py --agentdojo --model "$MODEL" --hpc \
    --attack_names template_ipi important_instructions \
    --defense_names spotlighting_delimiting spotlighting_datamarking spotlighting_encoding icon attention_tracker \
    --dedup_defended_clean \
    --only_samples "$SAMPLES" \
    --output_dir "$OUT"
echo "=== run exited $? at $(date) ==="

python3 analysis/repro_check.py compare "$ORIG" "$OUT" --samples "$SAMPLES" \
    --out "$R/analysis_out/repro_agentdojo_${MODEL}"
echo "RESULT: send analysis_out/repro_agentdojo_${MODEL}/report.md"
