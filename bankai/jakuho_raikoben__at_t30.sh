#!/bin/bash
# jakuho_raikoben__at_t30 -- Attention Tracker at 0.30 on AgentDojo: re-run only the band episodes
# ---------------------------------------------------------------------------
# Final run (P2a.1 code freeze). Resumable: if it hits the walltime, resubmit
# the SAME script; completed rows are skipped, errored rows retried.
# Blackwell nodes only (110/112): one GPU type per model, so a resume can never
# mix hardware (gpu_name is in run_settings and a mismatch is refused).
# Estimated: ~1-3 h per model
# Run AFTER the model's main AgentDojo job has finished its attention_tracker pass.
#   sbatch --export=ALL,MODEL=qwen3vl bankai/jakuho_raikoben__at_t30.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=jakuho_raikoben__at_t30
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster106,mscluster107,mscluster108,mscluster109,mscluster111
#SBATCH --nodes=1 --cpus-per-task=8 --mem=96G --time=3-00:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/jakuho_raikoben__at_t30_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/jakuho_raikoben__at_t30_%j.err

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
MODEL="${MODEL:-qwen3vl}"
OUT="$R/final_agentdojo_${MODEL}"
# --- code freeze guard: every row's code_version must show a clean commit ---
if git -C "$R" status --porcelain --untracked-files=all -- attacks defenses environments experiments extraction | grep -q '\.py$'; then
    echo "FATAL: uncommitted .py changes in the code directories -- commit first:" >&2
    git -C "$R" status --porcelain -- attacks defenses environments experiments extraction | grep '\.py$' >&2
    exit 1
fi
echo "code: $(git -C "$R" rev-parse --short HEAD)"

python3 analysis/at_threshold_band.py select "$OUT" --lo 0.30 --hi 0.50 > "$OUT/at_t30_samples.txt" || exit 1
if grep -q . "$OUT/at_t30_samples.txt"; then
    # Same flags as the main job (run settings must match), restricted to the band.
    python experiments/run_attack_matrix.py --agentdojo --model "$MODEL" --hpc \
        --attack_names template_ipi important_instructions \
        --defense_names attention_tracker_t30 \
        --dedup_defended_clean \
        --only_samples "$OUT/at_t30_samples.txt" \
        --output_dir "$OUT"
    echo "=== run exited $? ==="
else
    echo "nothing in the band: every 0.5 row is also the 0.3 row"
fi
mkdir -p "$R/analysis_out"
python3 analysis/at_threshold_band.py merge "$OUT" --lo 0.30 --hi 0.50 > "$R/analysis_out/at_t30_rows_${MODEL}.json"
