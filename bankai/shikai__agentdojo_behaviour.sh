#!/bin/bash
# shikai__agentdojo_behaviour -- CPU only, a few minutes. Behavioural results
# (ASR/UA with Wilson intervals, exposure and compliance bounds, Attention
# Tracker detection at every threshold, exact at 0.3 from the band merge, ICON flips) for the finished AgentDojo
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

rm -f "$OUT/report.md" "$OUT/numbers.json"     # a failed run must not leave the last report looking current
echo "== files (sha256, first 12)"
sha256sum analysis/agentdojo_behaviour.py tests/test_agentdojo_behaviour.py | cut -c1-12,65-
python3 -c "import agentdojo, importlib.metadata as m; print('agentdojo', m.version('agentdojo'))" 2>&1 | tail -1
echo "== fixture test"
if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest -q -p no:cacheprovider --tb=short tests/test_agentdojo_behaviour.py 2>&1 | grep -v "^\s*$" | tail -80
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RESULT: FAIL (fixture test) -- send this log"; exit 1; }
else
    echo "pytest not in this venv -- fixture test skipped (it passed locally)"
fi

# Attention Tracker at 0.3: the band merge written by jakuho_raikoben, when present
AT=()
for d in "${DIRS[@]}"; do
    b=$(basename "$d")
    f="analysis_out/at_t30_rows_${b#final_agentdojo_}.json"
    if [ -s "$f" ]; then AT+=("$b=$f"); echo "band merge: $f ($(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f") rows)"
    elif grep -q '"attention_tracker_t30"' "$d/attack_results.json"; then
        echo "FATAL: $d has attention_tracker_t30 rows (band episodes only) but no $f -- run the band merge" >&2; exit 1
    else echo "band merge: none for $b (no 0.3 results for it)"; fi
done
echo "== analysis: ${DIRS[*]} -> $OUT"
python3 analysis/agentdojo_behaviour.py "${DIRS[@]}" --out "$OUT" --at_rows ${AT[@]+"${AT[@]}"}
rc=$?
[ "$rc" -eq 0 ] && echo "RESULT: PASS -- send $OUT/report.md and $OUT/numbers.json" \
                || echo "RESULT: FAIL (exit $rc) -- send this log"
