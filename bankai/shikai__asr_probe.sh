#!/bin/bash
# shikai__asr_probe -- CPU only, read-only, a few minutes: template_ipi ASR by
# suite / template / injection task, exposure from the transcripts, for the
# running Gemma AgentDojo directory and any older Gemma AgentDojo directories.
#   sbatch bankai/shikai__asr_probe.sh
#SBATCH --job-name=shikai__asr_probe
#SBATCH --partition=bigbatch
#SBATCH --nodes=1 --cpus-per-task=2 --mem=8G --time=00:20:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_asr_probe_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_asr_probe_%j.out
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
DIRS=$(ls -d "$R"/final_agentdojo_gemma4 $(ls -dt "$R"/*agentdojo*gemma* 2>/dev/null | grep -v final_agentdojo_gemma4 | head -6) 2>/dev/null)
echo "directories: $DIRS"
python3 analysis/agentdojo_asr_probe.py $DIRS
