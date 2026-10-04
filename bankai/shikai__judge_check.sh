#!/bin/bash
# shikai__judge_check -- CPU only, ~5 min. Before the VWA final runs: does the
# dual-judge launcher import VisualWebArena, install and verify the judge
# patch on this venv? (job 63837 died at that step). Also reports which VWA
# import orders the venv's @beartype accepts, and the versions involved.
#   sbatch bankai/shikai__judge_check.sh
#SBATCH --job-name=shikai__judge_check
#SBATCH --partition=bigbatch
#SBATCH --nodes=1 --cpus-per-task=4 --mem=32G --time=00:20:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_judge_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_judge_%j.out
set -uo pipefail
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
source ~/vwa_adv_env.sh
export HF_HOME=/datasets/nmahomed/hf_cache
export AGENT_ATTACK_REPO=/datasets/nmahomed/vwa-adv/agent-attack
export OPENAI_API_KEY="$(cat ~/.openai_key)"     # imported by VWA's llms package; no call is made
echo "node: $(hostname)   code: $(git rev-parse --short HEAD)"
VWA=${VWA_ADV_REPO:-/datasets/nmahomed/vwa-adv/visualwebarena}

echo "== the checkout's beartype patch (first lines of browser_env/__init__.py)"
head -12 "$VWA/browser_env/__init__.py"
echo "== import orders (each in a fresh process)"
for order in "browser_env,evaluation_harness" "evaluation_harness.helper_functions" "agent.agent" "browser_env,agent.agent"; do
    python3 - "$VWA" "$order" <<'PY' 2>&1 | grep -E "^(OK|FAIL)"
import sys, importlib
sys.path.insert(0, sys.argv[1])
try:
    for m in sys.argv[2].split(","):
        importlib.import_module(m)
    print("OK  ", sys.argv[2])
except Exception as e:
    print("FAIL", sys.argv[2], "--", type(e).__name__, str(e)[:160])
PY
done

echo "== launcher check (what the VWA final runs do before the run)"
python3 experiments/run_with_dual_judge.py --vwa_path "$VWA" --dj_check_only --dj_no_local \
    --dj_records /dev/null 2>&1 | grep -E "dual_judge\]|launcher\]|FATAL|versions|Error"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && echo "RESULT: PASS -- the VWA final runs can start" \
                || echo "RESULT: FAIL (exit $rc) -- send this log"
