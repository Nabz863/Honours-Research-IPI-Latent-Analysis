#!/bin/bash
# shikai__backup -- CPU only. Copies every final result off the home directory
# to /datasets (the login node's disk is failing; cluster notice 16 Oct).
# Rows, transcripts, payload caches, judge records, analysis outputs, logs and
# the run scripts always; the activation files (.pt, large) unless --no-pt.
# rsync: a rerun copies only what changed, so run it again after new results.
#   sbatch bankai/shikai__backup.sh            (everything)
#   sbatch bankai/shikai__backup.sh --no-pt    (small files only)
#SBATCH --job-name=shikai__backup
#SBATCH --partition=bigbatch
#SBATCH --nodes=1 --cpus-per-task=2 --mem=8G --time=12:00:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_backup_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_backup_%j.out
set -uo pipefail
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
DEST=/datasets/nmahomed/backup/Honours-Research-IPI-Latent-Analysis
cd "$R"
echo "node: $(hostname)  at $(date)  code: $(git rev-parse --short HEAD)"
df -h "$R" /datasets/nmahomed | sed 's/^/  /'
mkdir -p "$DEST"
DIRS=()
for d in final_* repro_* analysis_out logs bankai strict_configs clean_configs; do [ -e "$d" ] && DIRS+=("$d"); done
# Pass 1: everything except activations, so the results are safe first even
# if the activations do not fit.
echo "== pass 1: rows, transcripts, payloads, judge records, analysis, logs"
for d in "${DIRS[@]}"; do
    rsync -a --exclude='*.pt' "$d" "$DEST/" || echo "   rsync FAILED for $d"
done
if [ "${1:-}" != "--no-pt" ]; then
    echo "== pass 2: activations"
    for d in "${DIRS[@]}"; do
        echo "   $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
        rsync -a "$d" "$DEST/" || echo "   rsync FAILED for $d (disk full? see df above)"
    done
fi
git bundle create "$DEST/repo_$(date +%Y%m%d).bundle" --all && echo "== git bundle written"
echo "== check: files in source vs backup (attack_results.json)"
for f in final_*/attack_results.json; do
    cmp -s "$f" "$DEST/$f" && echo "   identical: $f" || echo "   DIFFERS:   $f"
done
du -sh "$DEST" | sed 's/^/  backup size: /'
echo "RESULT: done at $(date)"
