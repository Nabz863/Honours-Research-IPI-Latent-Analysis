#!/bin/bash
# shikai__vwa_behaviour -- CPU only, about a minute. Behavioural results for the
# finished VWA-Adv run directories: rows that are not results (by kind), ASR and
# UA per attack x defence, published vs strict evaluator, modality groups,
# defences pooled with prevented/enabled pairs, textual exposure and compliance
# bounds. Read-only on the runs. Runs the fixture test first.
#   sbatch bankai/shikai__vwa_behaviour.sh final_vwa_gemma4 final_vwa_qwen3vl
#SBATCH --job-name=shikai__vwa_behaviour
#SBATCH --partition=bigbatch
#SBATCH --nodes=1 --cpus-per-task=2 --mem=16G --time=00:30:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_vwa_behaviour_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_vwa_behaviour_%j.out
set -uo pipefail
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
echo "node: $(hostname)   code: $(git rev-parse --short HEAD)"

DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=(final_vwa_gemma4 final_vwa_qwen3vl)
NAME=$(IFS=_; echo "${DIRS[*]}" | sed 's/final_vwa_//g')
OUT="analysis_out/vwa_behaviour_${NAME}"

rm -f "$OUT/report.md" "$OUT/numbers.json"     # a failed run must not leave the last report looking current
echo "== files (sha256, first 12)"
sha256sum analysis/vwa_behaviour.py tests/test_vwa_behaviour.py | cut -c1-12,65-
echo "== fixture test"
if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest -q -p no:cacheprovider --tb=short tests/test_vwa_behaviour.py 2>&1 | grep -v "^\s*$" | tail -60
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RESULT: FAIL (fixture test) -- send this log"; exit 1; }
else
    echo "pytest not in this venv -- fixture test skipped (it passed locally)"
fi

echo "== analysis: ${DIRS[*]} -> $OUT"
python3 analysis/vwa_behaviour.py "${DIRS[@]}" --out "$OUT"
rc=$?
[ "$rc" -eq 0 ] && echo "RESULT: PASS -- send $OUT/report.md and $OUT/numbers.json" \
                || echo "RESULT: FAIL (exit $rc) -- send this log"
