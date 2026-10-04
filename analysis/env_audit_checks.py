#!/usr/bin/env python3
"""
analysis/env_audit_checks.py -- environment-layer audit, the checks that need the
cluster. CPU only, no GPU, no browser, no model weights (tokenizers and a
meta-device model skeleton only). Every section is independent: one failing
prints its error and the rest still run.

Run via hpc_env_audit_checks.sh (bigbatch). Output is plain text for the
audit register.

 S1  VWA prompt template        -- is the task OBJECTIVE inside the text the
                                   spotlighting defence transforms?
 S2  VWA PromptAgent retry loop -- do parse-failure retries regenerate (and
                                   so append extra activation entries)?
 S3  VWA image captions         -- with captioning_fn=None, do non-victim
                                   images get any description?
 S4  tokenizer round trip       -- does decode(encode(obs)) inject a literal
                                   BOS string into the prompt?
 S5  Gemma 4 tool-call tokens   -- are Gemma's string delimiters special
                                   tokens that skip_special_tokens removes?
 S6  VWA-Adv task configs       -- eval types (stateful or not), require_reset,
                                   input images the runner never passes
 S7  observation truncation     -- does the victim image's line sit beyond
                                   max_obs_length=3840 tokens?
 S8  AgentDojo versions         -- task counts per benchmark version; the
                                   benchmark's own default system message
 S9  Gemma 4 module names       -- for the CrossInject projector lookup
 S10 AgentDojo no-injection ASR -- pairs whose injection task scores success
                                   with no payload present
 S11 VWA state leakage          -- attack success recorded where no attack
                                   was present (clean / defended-clean rows)
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
MODELS = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma4": "google/gemma-4-12B-it",
    "internvl3": "OpenGVLab/InternVL3-8B-hf",
    "llavanext": "llava-hf/llava-v1.6-mistral-7b-hf",
}
HF_CANDIDATES = list(dict.fromkeys(h for h in [
    "/datasets/nmahomed/hf_cache", os.environ.get("HF_HOME", ""),
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


def tokenizer(model_id):
    from transformers import AutoTokenizer
    errs = []
    for home in HF_CANDIDATES:
        try:
            return AutoTokenizer.from_pretrained(model_id, cache_dir=os.path.join(home, "hub"),
                                                 local_files_only=True)
        except Exception as e:
            errs.append(f"{home}: {type(e).__name__}")
    raise RuntimeError(f"{model_id} tokenizer not found in any cache: {errs}")


@section("S1  VWA prompt template (is the objective inside the defended text?)")
def s1():
    p = VWA / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"
    d = json.loads(p.read_text())
    print("template:\n" + d["template"])
    print("meta_data:", json.dumps(d.get("meta_data"), indent=1))
    print("\nconstruct() in the multimodal prompt constructor:")
    print(sh(f"grep -n 'template.format\\|objective=\\|max_obs_length\\|tokenizer.encode' "
             f"{VWA}/agent/prompts/prompt_constructor.py"))


@section("S2  VWA PromptAgent.next_action retry loop")
def s2():
    print(sh(f"grep -n 'def next_action\\|call_llm\\|max_retry\\|n += 1\\|ActionParsingError\\|force_prefix' "
             f"{VWA}/agent/agent.py"))


@section("S3  VWA image captions with captioning_fn=None")
def s3():
    print(sh(f"grep -rn 'captioning_fn\\|description:\\|adv_url2caption' {VWA}/browser_env/*.py | head -60"))


@section("S4  tokenizer decode(encode(x)) round trip -- BOS injected mid-prompt?")
def s4():
    sample = "[12] [A] [60 comments]\n[13] [IMG] [, description: a cat]"
    for short, mid in MODELS.items():
        try:
            t = tokenizer(mid)
        except Exception as e:
            print(f"{short:10s} SKIP: {e}")
            continue
        ids = t.encode(sample)
        rt = t.decode(ids)
        print(f"{short:10s} bos={t.bos_token!r} first_ids={ids[:3]} "
              f"round_trip_equal={rt == sample} prefix={rt[:len(rt) - len(sample)]!r}")


@section("S5  Gemma 4 tool-call special tokens")
def s5():
    t = tokenizer(MODELS["gemma4"])
    special = list(getattr(t, "all_special_tokens", [])) + \
        [str(x) for x in getattr(t, "additional_special_tokens", [])]
    added = {k: v for k, v in t.get_added_vocab().items()}
    hits = sorted({s for s in list(added) + special
                   if re.search(r"call|tool|escape|func|quote|\"", s, re.I)})
    print("special/added tokens mentioning call/tool/escape/func/quote:")
    for h in hits:
        print("  ", repr(h), added.get(h))
    tpl = getattr(t, "chat_template", None) or ""
    print("\nchat template mentions of tool-call syntax:")
    for m in re.finditer(r".{0,60}(call:|tool_call|escape).{0,60}", tpl):
        print("  ...", m.group(0).replace("\n", " "), "...")


@section("S6  VWA-Adv task configs: eval types, require_reset, input images")
def s6():
    strict = Path(os.environ.get("STRICT_ASR_CONFIG_DIR", R / "strict_configs"))
    clean = Path(os.environ.get("CLEAN_UA_CONFIG_DIR", R / "clean_configs"))
    rows = []
    for d in sorted(EXP.glob("agent_adv/reddit_*/")):
        if not (d / "data.json").exists():
            continue
        name = d.name
        adv = strict / f"{name}.json"
        adv = adv if adv.exists() else d / "config.json"
        if not adv.exists():
            continue
        c = json.loads(adv.read_text())
        base = "_".join(name.split("_")[:2])
        cc_path = clean / f"{name}.json"
        cc_path = cc_path if cc_path.exists() else EXP / "agent_clean" / base / "config.json"
        cc = json.loads(cc_path.read_text()) if cc_path.exists() else {}
        rows.append((name, base, c.get("eval", {}).get("eval_types"),
                     c.get("require_reset"), bool(c.get("image")),
                     cc.get("eval", {}).get("eval_types"), cc.get("require_reset"),
                     bool(cc.get("image")), adv.parent.name))
    print(f"{len(rows)} adversarial variants\n")
    print(f"{'task':40s} {'adv eval_types':40s} {'reset':5s} {'img':3s} | {'clean eval_types':30s} src")
    for r in rows:
        print(f"{r[0]:40s} {str(r[2]):40s} {str(r[3]):5s} {'Y' if r[4] else '-':3s} | "
              f"{str(r[5]):30s} {r[8]}")
    by_base = Counter(r[1] for r in rows)
    print("\nbase tasks shared by >1 variant (same Postmill objects, same run):")
    print({b: n for b, n in by_base.items() if n > 1})
    print("\nadv eval_types:", Counter(str(r[2]) for r in rows))
    print("clean eval_types:", Counter(str(r[5]) for r in rows))
    print("require_reset True (adv/clean):", sum(bool(r[3]) for r in rows), sum(bool(r[6]) for r in rows))
    print("configs with an input image (adv/clean):", sum(r[4] for r in rows), sum(r[7] for r in rows))


@section("S7  observation truncation at max_obs_length=3840 (recorded obs, Qwen tokenizer)")
def s7():
    t = tokenizer(MODELS["qwen3vl"])
    LIMIT = 3840
    over = 0
    n = 0
    for d in sorted(EXP.glob("agent_adv/reddit_*/")):
        dj = d / "data.json"
        if not dj.exists():
            continue
        data = json.loads(dj.read_text())
        vid = data.get("victim_som_id")
        base = "_".join(d.name.split("_")[:2])
        obs_p = EXP / "agent_clean" / base / "obs_text.txt"
        if vid is None or not obs_p.exists():
            print(f"{d.name:40s} no victim id / obs")
            continue
        obs = obs_p.read_text()
        lines = obs.splitlines()
        idx = next((i for i, l in enumerate(lines) if l.startswith(f"[{vid}] ")), None)
        if idx is None:
            print(f"{d.name:40s} victim [{vid}] not in recorded obs")
            continue
        upto = "\n".join(lines[:idx + 1])
        pos = len(t.encode(upto, add_special_tokens=False))
        tot = len(t.encode(obs, add_special_tokens=False))
        n += 1
        flag = "BEYOND LIMIT" if pos > LIMIT else ""
        over += pos > LIMIT
        print(f"{d.name:40s} victim line ends at token {pos:6d} of {tot:6d}  {flag}")
    print(f"\n{over}/{n} victim lines end beyond {LIMIT} tokens in the RECORDED observation.")
    print("Caveat: recorded with agent-attack's own viewport and captioner; ours is a "
          "1280x2048 full page (current_viewport_only=False) with no captioner, so "
          "positions differ. This sizes the risk; a live check settles it.")


@section("S8  AgentDojo: task counts per benchmark version; default system message")
def s8():
    from agentdojo.task_suite.load_suites import get_suites
    import agentdojo
    print("agentdojo", getattr(agentdojo, "__version__", "?"), agentdojo.__file__)
    for v in ["v1", "v1.1", "v1.1.1", "v1.1.2", "v1.2", "v1.2.1", "v1.2.2"]:
        try:
            s = get_suites(v)
            counts = {n: (len(x.user_tasks), len(x.injection_tasks)) for n, x in s.items()}
            pairs = sum(u * i for u, i in counts.values())
            print(f"  {v:8s} {counts}  pairs={pairs}")
        except Exception as e:
            print(f"  {v:8s} not available ({type(e).__name__})")
    try:
        from agentdojo.agent_pipeline.agent_pipeline import load_system_message
        print("\nload_system_message('default'):\n" + load_system_message(None))
    except Exception as e:
        print("load_system_message not importable:", type(e).__name__, e)
        print(sh("grep -rn 'def load_system_message' $(python3 -c 'import agentdojo,os;"
                 "print(os.path.dirname(agentdojo.__file__))')"))


@section("S9  Gemma 4 module names (meta device, no weights) + CrossInject projector paths")
def s9():
    import torch, transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    cfg = None
    for home in HF_CANDIDATES:
        try:
            cfg = AutoConfig.from_pretrained(MODELS["gemma4"], cache_dir=os.path.join(home, "hub"),
                                             local_files_only=True)
            break
        except Exception:
            continue
    if cfg is None:
        raise RuntimeError("gemma 4 config not in any cache")
    print("architectures:", cfg.architectures)
    cls = getattr(transformers, cfg.architectures[0])
    with init_empty_weights():
        m = cls(cfg)
    text_hidden = getattr(getattr(cfg, "text_config", cfg), "hidden_size", None)
    print("text hidden_size:", text_hidden)
    for name, mod in m.named_modules():
        if name.count(".") > 3:
            continue
        if re.search(r"proj|vision|embed|connector|mm_|multi_modal|adapter", name, re.I):
            shape = ""
            w = getattr(mod, "weight", None)
            if w is not None:
                shape = tuple(w.shape)
            print(f"  {name:70s} {type(mod).__name__:35s} {shape}")
    print("\n" + sh(f"grep -n '_PROJECTOR_PATHS' -A 14 {R}/attacks/cross_modal_injection.py"))


@section("S10 AgentDojo: injection-task success with NO payload present")
def s10():
    for d in ("agentdojo_matrix_v4_qwen3vl", "agentdojo_matrix_v4_gemma4_v3"):
        p = R / d / "attack_results.json"
        if not p.exists():
            print(d, "missing"); continue
        rows = json.load(open(p))
        clean_hit = sorted(r["task_id"] for r in rows
                           if r.get("condition") == "clean" and r.get("asr_success") is True
                           and r.get("error") is None)
        print(f"{d}: {len(clean_hit)} CLEAN episodes (no payload anywhere) score injection success")
        for t in clean_hit:
            print("   ", t)
        succ = defaultdict(set)
        for r in rows:
            if r.get("condition") == "attacked" and r.get("asr_success") is True and r.get("error") is None:
                succ[r["task_id"]].add(r["attack_name"])
        allthree = sorted(t for t, a in succ.items() if len(a) >= 3)
        print(f"  pairs where >=3 attacks all succeed: {allthree}")


@section("S11 VWA state leakage: attack success where no attack was present")
def s11():
    for p in sorted(glob.glob(str(R / "vwa_adv_*" / "attack_results.json"))):
        rows = json.load(open(p))
        d = Path(p).parent.name
        prior_success = defaultdict(int)       # base_task -> attacked successes so far
        clean_rows = leak_rows = 0
        report = []
        by_cond = defaultdict(lambda: [0, 0])
        for i, r in enumerate(rows):
            if r.get("error") is not None or r.get("asr_success") is None:
                continue
            cond = r.get("condition")
            base = r.get("base_task") or r.get("task_id")
            by_cond[cond][0] += bool(r["asr_success"]); by_cond[cond][1] += 1
            if cond in ("clean", "defended_clean"):
                clean_rows += 1
                if r["asr_success"]:
                    leak_rows += 1
                    report.append(f"    row {i:5d} {r['sample_id']:60s} strict={r.get('strict_asr')} "
                                  f"prior attacked successes on {base}: {prior_success[base]}")
            elif r["asr_success"]:
                prior_success[base] += 1
        print(f"\n{d}: {leak_rows}/{clean_rows} no-attack rows score attack success")
        print("   ASR by condition:", {k: f"{v[0]}/{v[1]}" for k, v in by_cond.items()})
        for line in report[:40]:
            print(line)
        if len(report) > 40:
            print(f"    ... {len(report) - 40} more")


if __name__ == "__main__":
    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11):
        fn()
    print("\nALL SECTIONS ATTEMPTED")
