#!/bin/bash
#SBATCH --job-name=geom_rerun
#SBATCH --partition=stampede
#SBATCH --nodes=1 --cpus-per-task=4 --mem=16G --time=04:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/geom_rerun_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/geom_rerun_%j.err
set -uo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
for M in qwen3vl internvl llava; do
  python analysis/geometry.py --results_dir agentdojo_matrix_v4_$M \
      --out_dir analysis_out/${M}_v4_clean --turn_strategy mean
done
python analysis/geometry.py --results_dir agentdojo_matrix_v4_gemma4_v2 \
    --out_dir analysis_out/gemma4_v4_clean --turn_strategy mean
python analysis/geometry.py --results_dir vwa_adv_qwen3vl \
    --out_dir analysis_out/vwa_qwen3vl_clean --turn_strategy mean
