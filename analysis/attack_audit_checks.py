#!/usr/bin/env python3
"""
attack_audit_checks.py -- the attack-audit checks that need the cluster.
CPU only: processors, tokenizers, meta-device skeletons, JSON. No GPU, no
browser, no weights. Each section is independent.

 A1 victim images: stored size vs the size the page renders them at
 A2 how adv_url2image is delivered into the page (size attributes, encoding)
 A3 geometry-adapter gate (verify_adapter + check_gradients) -- Qwen3-VL,
    Gemma 4 -- on real victim images
 A4 GCG suffix re-tokenisation: does the optimised token sequence survive
    decode -> re-encode?
 A5 layers: which layer obfuscated_activation targets vs the layer analysed
 A6 template_ipi: which template each task got, per model directory
 A7 cross_modal_delivery_check.py -- what it verified
 A8 the real adversarial goals and target labels; COMET entity extraction
    on each (does the entangled text still carry the goal?)
 A9 what the attacks actually did in existing runs, from row metadata
    (GCG loss, CrossInject VLA loss start/end, typographic font/truncation,
    steganographic round trip, COMET leakage)
"""

import glob
import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

R = Path(os.environ.get("R", "/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis"))
VWA = Path(os.environ.get("VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
AA = Path(os.environ.get("AGENT_ATTACK_REPO", "/datasets/nmahomed/vwa-adv/agent-attack"))
EXP = AA / "exp_data"
sys.path.insert(0, str(R))
MODELS = {"qwen3vl": "Qwen/Qwen3-VL-8B-Instruct", "gemma4": "google/gemma-4-12B-it"}
HF = list(dict.fromkeys(h for h in ["/datasets/nmahomed/hf_cache", os.environ.get("HF_HOME", ""),
                                     os.path.expanduser("~/.cache/huggingface")] if h))


def section(name):
    def deco(fn):
        def run():
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
            try:
                fn()
            except Exception:
                print("SECTION FAILED:")
                traceback.print_exc(file=sys.stdout)
        return run
    return deco


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).rstrip()


def load(kind, mid):
    import transformers
    cls = getattr(transformers, kind)
    for h in HF:
        try:
            return cls.from_pretrained(mid, cache_dir=os.path.join(h, "hub"), local_files_only=True)
        except Exception:
            continue
    raise RuntimeError(f"{kind} for {mid} not found in {HF}")


def victim_dirs(n=None):
    ds = [d for d in sorted(EXP.glob("agent_adv/reddit_*/"))
          if (d / "data.json").exists() and (d / "victim_image.png").exists()]
    return ds[:n] if n else ds


@section("A1 victim images: stored size vs rendered size")
def a1():
    from PIL import Image
    sizes = Counter()
    for d in victim_dirs():
        data = json.loads((d / "data.json").read_text())
        im = Image.open(d / "victim_image.png")
        pos = data.get("position") or {}
        print(f"{d.name:40s} png={im.size}  position={json.dumps(pos)[:90]}")
        sizes[im.size] += 1
    print("\npng size distribution:", sizes.most_common(10))
    first = victim_dirs(1)
    if first:
        base = "_".join(first[0].name.split("_")[:2])
        print("\nfiles in", first[0]); print(sh(f"ls -la {first[0]}"))
        print("\nfiles in agent_clean/" + base); print(sh(f"ls -la {EXP}/agent_clean/{base}"))


@section("A2 how adv_url2image is delivered into the page")
def a2():
    for root in (VWA, AA):
        print(f"--- {root}")
        print(sh(f"grep -rn 'adv_url2image' {root} --include=*.py | head -30"))
    print("\n--- the substitution itself (context):")
    print(sh(f"grep -rn -A25 'def .*adv_url2image\\|adv_url2image\\[' {VWA}/browser_env --include=*.py | head -120"))


@section("A3 geometry-adapter gate on real victim images")
def a3():
    from PIL import Image
    from attacks.geometry_adapters import verify_adapter, check_gradients
    imgs = [Image.open(d / "victim_image.png").convert("RGB") for d in victim_dirs(3)]
    for short, mid in MODELS.items():
        try:
            proc = load("AutoProcessor", mid)
        except Exception as e:
            print(short, "SKIP", e); continue
        for im in imgs:
            ok, rep = verify_adapter(proc, im, verbose=False)
            print(f"{short:8s} {str(im.size):12s} verify={'PASS' if ok else 'FAIL'} "
                  f"shape={rep.get('got_shape')} ref={rep.get('ref_shape')} "
                  f"max_err={rep.get('max_abs_err')} grid_match={rep.get('image_grid_thw_match')}")
        print(f"{short:8s} gradients reach image: {check_gradients(proc, imgs[0], verbose=False)}")


@section("A4 GCG suffix re-tokenisation")
def a4():
    for short, mid in MODELS.items():
        try:
            tok = load("AutoTokenizer", mid)
        except Exception as e:
            print(short, "SKIP", e); continue
        seed = tok.encode("x", add_special_tokens=False)[-1]
        ids = [seed] * 20
        txt = tok.decode(ids)
        re_ids = tok.encode(txt, add_special_tokens=False)
        print(f"{short}: seed suffix 20 x {tok.decode([seed])!r} -> text {txt!r} -> "
              f"{len(re_ids)} tokens on re-encode (identical={re_ids == ids})")
        print(f"{short}: encode('Send money') with special tokens -> "
              f"{tok.convert_ids_to_tokens(tok.encode('Send money'))}")
    # stored payloads: does the optimised suffix survive?
    for d in sorted(glob.glob(str(R / "agentdojo_matrix_v4_*" / "adv_payloads_cache.json")))[:2]:
        rows = json.load(open(d))
        g = [r for r in rows if r.get("attack_name") == "gcg"][:1]
        print(f"\n{Path(d).parent.name}: first gcg payload tail: {g[0]['adversarial_text'][-160:]!r}"
              if g else f"{d}: no gcg rows")


@section("A5 obfuscated_activation target layer vs analysis layer")
def a5():
    import torch, transformers
    from accelerate import init_empty_weights
    from extraction.hooks import _resolve_target_layers
    for short, mid in MODELS.items():
        cfg = load("AutoConfig", mid)
        with init_empty_weights():
            m = getattr(transformers, cfg.architectures[0])(cfg)
        tl = list(_resolve_target_layers(m, None))
        print(f"{short}: target_layers={tl}")
        print(f"   curated VWA runner: reference taken at target_layers[len//2] = {tl[len(tl)//2]}, "
              f"attack optimises target_layers[0] = {tl[0]}  -> {'MISMATCH' if tl[0] != tl[len(tl)//2] else 'same'}")
        print(f"   AgentDojo runner default (no --obfuscation_target_layer): {tl[0]}; "
              f"analysis layer: deepest captured = {tl[-1]}")


@section("A6 template_ipi: template per task, per directory")
def a6():
    per = defaultdict(dict)
    for p in sorted(glob.glob(str(R / "*" / "attack_results.json"))):
        try:
            rows = json.load(open(p))
        except Exception:
            continue
        d = Path(p).parent.name
        for r in rows:
            if r.get("attack_name") == "template_ipi" and r.get("condition") == "attacked":
                t = r.get("attack_meta_template")
                if t:
                    per[d][r["task_id"]] = t
    names = sorted(per)
    for d in names:
        c = Counter(per[d].values())
        print(f"{d:45s} {len(per[d]):4d} tasks  templates used: {[(k[:25], v) for k, v in c.items()]}")
    for a in names:
        for b in names:
            if a < b:
                common = set(per[a]) & set(per[b])
                if common:
                    diff = sum(per[a][t] != per[b][t] for t in common)
                    print(f"   {a} vs {b}: {diff}/{len(common)} shared tasks got a different template")


@section("A7 cross_modal_delivery_check.py")
def a7():
    p = R / "experiments" / "cross_modal_delivery_check.py"
    print(sh(f"sed -n 1,80p {p}"))
    print(sh(f"ls -la {R}/logs | grep -i deliver | tail -5"))


@section("A8 real goals, target labels, COMET extraction")
def a8():
    from attacks.comet import _extract_entities
    for d in victim_dirs():
        data = json.loads((d / "data.json").read_text())
        goal = data.get("adversarial_goal") or ""
        ents, rej = _extract_entities(goal, 4) if goal else ([], [])
        print(f"{d.name:40s} target={data.get('target_label')!r}")
        print(f"    goal    : {goal!r}")
        print(f"    COMET   : entities={ents} rejected={rej}")


@section("A9 what the attacks did in existing runs (row metadata)")
def a9():
    keys = {
        "gcg": ["attack_meta_num_steps", "attack_meta_num_restarts", "attack_meta_final_loss"],
        "obfuscated_activation": ["attack_meta_num_steps", "attack_meta_target_layer",
                                  "attack_meta_final_behavioral_ce_loss",
                                  "attack_meta_final_geometry_distance"],
        "cross_modal_injection": ["attack_meta_projector_path", "attack_meta_vla_loss_start",
                                  "attack_meta_vla_loss_end", "attack_meta_ce_loss_end"],
        "typographic": ["attack_meta_truetype_font", "attack_meta_font_size",
                        "attack_meta_text_truncated", "attack_meta_upscaled_from"],
        "steganographic": ["attack_meta_raw_roundtrip_ok", "attack_meta_repetitions_embedded"],
        "comet": ["attack_meta_n_hops_realised", "attack_meta_goal_leaked_in_text",
                  "attack_meta_render"],
        "adversarial_patch": ["attack_meta_final_loss"],
        "pixel_perturbation": ["attack_meta_iters", "attack_meta_adapter"],
    }
    for p in sorted(glob.glob(str(R / "vwa_adv_*" / "attack_results.json"))) + \
             sorted(glob.glob(str(R / "agentdojo_matrix_v4_*" / "attack_results.json"))):
        try:
            rows = [r for r in json.load(open(p))
                    if r.get("condition") == "attacked" and r.get("error") is None]
        except Exception as e:
            print(p, type(e).__name__); continue
        if not rows:
            continue
        print(f"\n{Path(p).parent.name}  ({len(rows)} attacked rows)")
        by = defaultdict(list)
        for r in rows:
            by[r.get("attack_name")].append(r)
        for atk, rs in sorted(by.items()):
            for k in keys.get(atk, []):
                vals = [r.get(k) for r in rs]
                nums = sorted(v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool))
                if len(set(nums)) > 3:
                    print(f"  {atk:22s} {k:40s} n={len(nums)} min={nums[0]:.4g} "
                          f"med={nums[len(nums)//2]:.4g} max={nums[-1]:.4g}")
                else:
                    c = Counter(v if isinstance(v, str) else json.dumps(v)[:80] for v in vals)
                    print(f"  {atk:22s} {k:40s} {dict(c.most_common(4))}")
            if atk == "cross_modal_injection":
                dec = sum(1 for r in rs
                          if isinstance(r.get("attack_meta_vla_loss_start"), (int, float))
                          and isinstance(r.get("attack_meta_vla_loss_end"), (int, float))
                          and r["attack_meta_vla_loss_end"] < r["attack_meta_vla_loss_start"])
                print(f"  {atk:22s} VLA loss decreased in {dec}/{len(rs)} rows")


if __name__ == "__main__":
    for fn in (a1, a2, a3, a4, a5, a6, a7, a8, a9):
        fn()
    print("\nALL SECTIONS ATTEMPTED")
