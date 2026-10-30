#!/usr/bin/env python3
"""analysis/logit_lens.py -- what the injection direction says in vocabulary
space (RQ1). CPU only, read-only on the run directory and the model files.

For every reached undefended (clean, attacked) pair, at the payload-arrival
step (geometry.arrival_or_acts, the same pairing as every other analysis) and
the analysis layer L, the two residual-stream states are passed through the
model's own final RMSNorm and unembedding (the logit lens). Because the lens is
linear after the norm, the mean change in lens logits over pairs is
    W_U (mean n(h_adv) - mean n(h_clean)),
reported as the tokens whose logits the injection raises and lowers most:
over all reached pairs, over successful and over failed attacks, and the
successful-minus-failed contrast.

Correctness check before any result: the norm convention (gain w or 1 + w) and
the tensor names are read from the weights, and the lens at the LAST layer of
clean episodes must reproduce the model's own next token (the first token of
the completion recorded at that step) -- greedy decoding makes that exact when
the keys and the convention are right. Both conventions are scored and the one
that reproduces the model is used; the match rates are printed.

  python3 analysis/logit_lens.py --results_dir final_agentdojo_gemma4 \
      --model_dir /datasets/nmahomed/hf_cache/hub/models--google--gemma-4-12B-it/snapshots/<hash> \
      --layer 42 --out analysis_out/logit_lens_agentdojo_gemma4.json
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import geometry as g  # noqa: E402

_NOT_TEXT = re.compile(r"vision|visual|audio|image|multi_modal|mm_|projector|per_layer|layers?\.\d")


def text_config(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config") or cfg
    return cfg, tc


def weight_files(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(model_dir, f) for k, f in wm.items()}
    from safetensors import safe_open
    f = os.path.join(model_dir, "model.safetensors")
    with safe_open(f, framework="pt") as fh:
        return {k: f for k in fh.keys()}


def pick(keys, suffix):
    c = [k for k in keys if k.endswith(suffix) and not _NOT_TEXT.search(k)]
    return c


def load_tensor(files, key):
    from safetensors import safe_open
    with safe_open(files[key], framework="pt") as fh:
        return fh.get_tensor(key).float()


def lens_weights(model_dir):
    """(W_U [V, d] float32, norm weight [d], eps, keys used)."""
    files = weight_files(model_dir)
    keys = sorted(files)
    unemb = pick(keys, "lm_head.weight") or pick(keys, "embed_tokens.weight")
    norm = [k for k in pick(keys, "norm.weight") if re.search(r"(^|\.)norm\.weight$", k)]
    if len(unemb) != 1 or len(norm) != 1:
        raise SystemExit(f"FATAL: ambiguous lens tensors: unembedding {unemb}, final norm {norm}")
    _, tc = text_config(model_dir)
    eps = float(tc.get("rms_norm_eps", 1e-6))
    return load_tensor(files, unemb[0]), load_tensor(files, norm[0]), eps, {"unembedding": unemb[0], "final_norm": norm[0]}


def rmsnorm(x, gain, eps):
    x = torch.as_tensor(x, dtype=torch.float32)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain


def reached_pairs(results_dir, attack, layers):
    """[(asr, t, clean_sid, {L: h_clean}, {L: h_adv})] for reached undefended pairs."""
    rows = json.load(open(os.path.join(results_dir, "attack_results.json")))
    clean = {r["task_id"]: r["sample_id"] for r in rows
             if r.get("condition") == "clean" and not r.get("error")}
    plan, needs = [], defaultdict(set)
    for r in rows:
        if (r.get("condition") != "attacked" or r.get("error") or r.get("attack_name") != attack
                or r.get("defense_name") not in (None, "none")):
            continue
        c = clean.get(r["task_id"])
        if c is None:
            continue
        t, st = g.arrival_or_acts(results_dir, c, r["sample_id"])
        if t is None:
            continue
        plan.append((g.normalise_asr(r), t, c, r["sample_id"]))
        needs[c].add(t)
        needs[r["sample_id"]].add(t)
    v = g.load_turns(results_dir, needs, layers)
    out = []
    for asr, t, c, a in plan:
        hc, ha = v.get((c, t)), v.get((a, t))
        if hc and ha and all(L in hc and L in ha for L in layers):
            out.append((asr, t, c, hc, ha))
    return out


def first_token(tok, text):
    ids = tok.encode(text or "", add_special_tokens=False).ids
    return ids[0] if ids else None


def stored_layers(results_dir, sid):
    """Layer keys stored in one episode's .pt (first turn)."""
    turns = torch.load(os.path.join(results_dir, sid + ".pt"), map_location="cpu", weights_only=True)
    return sorted(int(k) for k in turns[0])


def check_convention(results_dir, pairs, last, W, w, eps, tok, n_max=80):
    """Top-1 / top-5 rate at which the lens on the LAST layer of clean states
    reproduces the recorded completion's first token, per gain convention."""
    from experiments.attention_replay import read_transcript
    seen, items = set(), []
    for _, t, c, hc, _ in pairs:
        if (c, t) in seen or last not in hc:
            continue
        seen.add((c, t))
        tr = read_transcript(Path(results_dir) / f"{c}.transcript.json.gz")
        if not tr or t >= len(tr["steps"]):
            continue
        ft = first_token(tok, tr["steps"][t].get("completion"))
        if ft is not None:
            items.append((hc[last], ft))
        if len(items) >= n_max:
            break
    res = {}
    for name, gain in (("w", w), ("1+w", 1.0 + w)):
        top1 = top5 = 0
        for h, ft in items:
            logits = W @ rmsnorm(h, gain, eps)
            top = torch.topk(logits, 5).indices.tolist()
            top1 += top[0] == ft
            top5 += ft in top
        res[name] = {"n": len(items), "top1": top1 / max(len(items), 1), "top5": top5 / max(len(items), 1)}
    return res


def top_tokens(W, v, tok, k):
    logits = W @ torch.as_tensor(v, dtype=torch.float32)
    up = torch.topk(logits, k)
    dn = torch.topk(-logits, k)
    dec = lambda i: tok.decode([int(i)])
    return {"up": [[dec(i), float(s)] for s, i in zip(up.values, up.indices)],
            "down": [[dec(i), float(-s)] for s, i in zip(dn.values, dn.indices)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--attacks", nargs="+", default=["important_instructions", "template_ipi"])
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    W, w, eps, keys = lens_weights(a.model_dir)
    _, tc = text_config(a.model_dir)
    last = int(tc["num_hidden_layers"]) - 1
    print(f"lens tensors: {keys}; W_U {tuple(W.shape)}; eps {eps}; analysis layer {a.layer}; last layer {last}")
    report = {"config": vars(a), "keys": keys, "last_layer": last, "attacks": {}}
    rows = json.load(open(os.path.join(a.results_dir, "attack_results.json")))
    c0 = next(r["sample_id"] for r in rows if r.get("condition") == "clean" and not r.get("error"))
    have = stored_layers(a.results_dir, c0)
    # the lens needs the residual stream BEFORE the final norm at the last decoder
    # layer: keys must be decoder-layer outputs 0..N-1. A key N would mean
    # hidden_states indexing (N = already normalised) -- refuse rather than guess.
    if max(have) >= last + 1:
        raise SystemExit(f"FATAL: stored layer keys reach {max(have)} with {last + 1} decoder layers: "
                         "hidden_states indexing? The lens would be applied to the wrong state.")
    if last not in have or a.layer not in have:
        raise SystemExit(f"FATAL: layer {last} (last) or {a.layer} (analysis) not stored; stored {have[0]}..{have[-1]}")
    gain = None
    for attack in a.attacks:
        pairs = reached_pairs(a.results_dir, attack, [a.layer, last])
        print(f"\n== {attack}: {len(pairs)} reached pairs with both layers")
        if not pairs:
            report["attacks"][attack] = {"note": "no reached pairs"}
            continue
        if gain is None:
            chk = check_convention(a.results_dir, pairs, last, W, w, eps, tok)
            n = chk["w"]["n"]
            score = {k: (v["top1"], v["top5"]) for k, v in chk.items()}
            report["convention_check"] = chk
            if n < 20 or score["w"] == score["1+w"]:
                raise SystemExit(f"FATAL: the norm convention cannot be established ({n} checks, scores {score}); "
                                 "no token list is written")
            best = max(score, key=score.get)
            gain = w if best == "w" else 1.0 + w
            report["convention"] = best
            print(f"   convention check (last-layer lens vs the model's own next token): {chk} -> using gain = {best}")
            if chk[best]["top5"] < 0.5:
                report["warning"] = "lens reproduces the model's next token (top-5) in fewer than half the checks"
                print(f"   WARNING: {report['warning']}; treat the token lists below with caution")
        groups = {"all": pairs, "success": [p for p in pairs if p[0] is True],
                  "failure": [p for p in pairs if p[0] is False]}
        mean_d = {}
        ent = {"n": {k: len(v) for k, v in groups.items()}}
        for name, ps in groups.items():
            if not ps:
                continue
            d = torch.stack([rmsnorm(p[4][a.layer], gain, eps) - rmsnorm(p[3][a.layer], gain, eps) for p in ps]).mean(0)
            mean_d[name] = d
            ent[name] = top_tokens(W, d, tok, a.k)
        if "success" in mean_d and "failure" in mean_d:
            ent["success_minus_failure"] = top_tokens(W, mean_d["success"] - mean_d["failure"], tok, a.k)
        report["attacks"][attack] = ent
        for name in ("all", "success", "failure", "success_minus_failure"):
            if name in ent:
                print(f"   [{name}] raised: " + ", ".join(repr(t) for t, _ in ent[name]["up"][:15]))
                print(f"   [{name}] lowered: " + ", ".join(repr(t) for t, _ in ent[name]["down"][:15]))
    if gain is None:
        raise SystemExit("FATAL: no attack had reached pairs with both layers; nothing written")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    json.dump(report, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
