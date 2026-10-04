#!/bin/bash
# shikai__node_check -- can a node take VWA (fakeroot, bf16 GPU, Chromium, Postmill)?
#   sbatch -p biggpu -w mscluster110 bankai/shikai__node_check.sh
#SBATCH --job-name=shikai__node_check
#SBATCH --nodes=1 --cpus-per-task=4 --mem=32G --time=00:15:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/shikai_%j.out
R=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
cd "$R"; source "$R/venv/bin/activate"
echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
grep -q "^$(id -un):" /etc/subuid && echo "PASS subuid" || echo "FAIL subuid (no fakeroot mapping)"
python3 - <<'PY'
import time, torch
ok = torch.cuda.is_available(); print("PASS cuda" if ok else "FAIL cuda")
if ok:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize()
    t = time.time()
    for _ in range(50): a @ a
    torch.cuda.synchronize(); print(f"bf16 matmul: {50*2*4096**3/(time.time()-t)/1e12:.1f} TFLOP/s (Blackwell ~100+, RTX 8000 ~10)")
from playwright.sync_api import sync_playwright
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True); b.close(); print("PASS chromium")
except Exception as e: print("FAIL chromium:", str(e)[:150])
PY
mkdir -p /var/tmp/nmahomed_apptainer_run && export APPTAINER_TMPDIR=/var/tmp/nmahomed_apptainer_run
apptainer instance start --fakeroot --no-home --writable-tmpfs /datasets/nmahomed/containers/sif/postmill.sif shikai_probe \
  && echo "PASS postmill container" && apptainer instance stop shikai_probe || echo "FAIL postmill container"
