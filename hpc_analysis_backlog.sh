#!/bin/bash
#SBATCH --job-name=analysis_backlog
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=06:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/backlog_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/backlog_%j.err
set -uo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis

for A in template_ipi typographic steganographic comet gcg; do
  echo "########## RQ2 VWA-Adv, attack=$A"
  python analysis/rq2_geometry.py --results_dir vwa_adv_qwen3vl_defended \
      --layer 31 --attack "$A" --out "analysis_out/rq2_vwa_${A}.json"
done

echo "########## RQ2 AgentDojo (recheck, all four models)"
for M in qwen3vl internvl llava; do
  python analysis/rq2_geometry.py --results_dir "agentdojo_matrix_v4_${M}" \
      --layer 31 --attack template_ipi --out "analysis_out/rq2_${M}.json"
done

echo "########## geometry: Gemma AgentDojo"
python analysis/geometry.py --results_dir agentdojo_matrix_v4_gemma4_v2 \
    --out_dir analysis_out/gemma4_v4 --turn_strategy mean

echo "########## stats: Gemma AgentDojo (layer 42 -- its band is 24-42)"
python analysis/stats.py --results_dir agentdojo_matrix_v4_gemma4_v2 \
    --layer 42 --n_perm 500 --out analysis_out/stats_gemma4_v4.json
