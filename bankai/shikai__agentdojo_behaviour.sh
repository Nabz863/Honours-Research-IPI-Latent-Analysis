#!/bin/bash
# shikai__agentdojo_behaviour -- CPU only, a few minutes. Behavioural results
# (ASR/UA with Wilson intervals, exposure and compliance bounds, Attention
# Tracker detection at every threshold, ICON flips) for the finished AgentDojo
# run directories. Read-only on the runs. Runs the fixture test first.
#   sbatch bankai/shikai__agentdojo_behaviour.sh                       (Gemma only)
#   sbatch bankai/shikai__agentdojo_behaviour.sh final_agentdojo_gemma4 final_agentdojo_qwen3vl
#SBATCH --job-name=shikai__agentdojo_behaviour
#SBATCH --partition=bigbatch
#SBATCH --nodes=1 --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_behaviour_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_behaviour_%j.out
set -uo pipefail
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
export HF_HOME=/datasets/nmahomed/hf_cache
export AGENTDOJO_BENCHMARK_VERSION=v1.1.2
echo "node: $(hostname)   code: $(git rev-parse --short HEAD)"

DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=(final_agentdojo_gemma4)
NAME=$(IFS=_; echo "${DIRS[*]}" | sed 's/final_agentdojo_//g')
OUT="analysis_out/agentdojo_behaviour_${NAME}"

echo "== fixture test"
if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest -q -p no:cacheprovider tests/test_agentdojo_behaviour.py 2>&1 | tail -3
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RESULT: FAIL (fixture test) -- send this log"; exit 1; }
else
    echo "pytest not in this venv -- fixture test skipped (it passed locally)"
fi

echo "== analysis: ${DIRS[*]} -> $OUT"
python3 analysis/agentdojo_behaviour.py "${DIRS[@]}" --out "$OUT"
rc=$?
[ "$rc" -eq 0 ] && echo "RESULT: PASS -- send $OUT/report.md and $OUT/numbers.json" \
                || echo "RESULT: FAIL (exit $rc) -- send this log"
