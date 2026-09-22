#!/bin/bash
#SBATCH --job-name=gemma_smoke
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster110,mscluster112
#SBATCH --nodes=1 --cpus-per-task=8 --mem=48G --time=00:40:00
#SBATCH --output=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/gemma_smoke_%j.out
#SBATCH --error=/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/gemma_smoke_%j.err
set -euo pipefail
source /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/venv/bin/activate
export HF_HOME=/datasets/nmahomed/hf_cache TRANSFORMERS_CACHE=$HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis
python - <<'PY'
import torch
from extraction.model_loader import load_gemma4_12b
from extraction.hooks import ActivationExtractor, _resolve_target_layers, _get_layer_list
m, p = load_gemma4_12b(hpc=True)
print("layers found:", len(_get_layer_list(m)))
tl = _resolve_target_layers(m, None)
print("target layers:", tl[0], "..", tl[-1], f"({len(tl)} layers)")
msgs = [{"role":"user","content":[{"type":"text","text":"Reply with the single word: ready"}]}]
txt = p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inp = p(text=txt, return_tensors="pt").to(m.device)
ext = ActivationExtractor(m, target_layers=tl, token_aggregation="last")
with ext:
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=16, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
acts = ext.get_activations()
print("completion:", repr(p.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)))
print("captured layers:", len(acts), "| example dim:", list(acts.values())[0].shape)
PY
