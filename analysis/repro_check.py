#!/usr/bin/env python3
"""
analysis/repro_check.py -- run-to-run reproducibility of the final AgentDojo
runs (Chapter 3, "Determinism": two identical runs compared on outcomes and
activations).

    select   picks a fixed sample of episodes from a finished run directory
             and writes their sample ids, one per line, for
             run_attack_matrix.py --only_samples
    compare  compares the re-run directory with the original, episode by
             episode: scores, payloads, every step's prompt hash and
             completion, tool calls, and the saved activations (every turn,
             every layer)

    python3 analysis/repro_check.py select final_agentdojo_gemma4 > repro_samples.txt
    python3 analysis/repro_check.py compare final_agentdojo_gemma4 repro_agentdojo_gemma4 \
        --out analysis_out/repro_agentdojo_gemma4

SAMPLE (fixed in advance, seeded): 10 user tasks stratified by suite
(workspace 3, travel 2, banking 2, slack 3); for each, one injection task,
skipping pairs excluded from the analysis (clean episode scores attack
success); for each pair five episodes: clean, the two attacks undefended,
important_instructions under spotlighting delimiting and under ICON. Only
pairs whose five original episodes all completed are eligible.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PER_SUITE = {"workspace": 3, "travel": 2, "banking": 2, "slack": 3}
SEED = 20261015


def sids_for(tid: str) -> List[str]:
    return [f"clean_{tid}", f"important_instructions_{tid}", f"template_ipi_{tid}",
            f"important_instructions_defended_spotlighting_delimiting_{tid}",
            f"important_instructions_defended_icon_{tid}"]


def load_rows(d: Path) -> Dict[str, Dict[str, Any]]:
    return {r["sample_id"]: r for r in json.loads((d / "attack_results.json").read_text())}


def select(d: Path) -> List[str]:
    rows = load_rows(d)
    ok = {s for s, r in rows.items() if not r.get("error")}
    pairs: Dict[tuple, List[tuple]] = {}
    for s, r in rows.items():
        if r.get("condition") != "clean" or r.get("error"):
            continue
        if r.get("asr_success") is True:                 # excluded pair
            continue
        tid = r["task_id"]
        if all(x in ok for x in sids_for(tid)):
            pairs.setdefault((r["suite_name"], r["user_task_id"]), []).append((r["injection_task_id"], tid))
    rng = random.Random(SEED)
    out = []
    for suite, k in PER_SUITE.items():
        users = sorted(u for (s, u) in pairs if s == suite)
        if len(users) < k:
            sys.exit(f"FATAL: only {len(users)} eligible {suite} user tasks, need {k}")
        for u in rng.sample(users, k):
            _, tid = rng.choice(sorted(pairs[(suite, u)]))
            out += sids_for(tid)
    return out


def _turns_text(tr: Optional[Dict[str, Any]]):
    if not tr:
        return None
    return [(s.get("prompt_sha256"), s.get("completion")) for s in tr.get("steps", [])]


def _read_tr(p: Path):
    try:
        return json.loads(gzip.decompress(p.read_bytes()))
    except Exception:
        return None


def _act_diff(a, b) -> Dict[str, Any]:
    """Largest absolute and relative difference over every turn and layer."""
    import torch
    if len(a) != len(b):
        return {"turns": [len(a), len(b)], "comparable": False}
    max_abs = max_rel = 0.0
    n = 0
    for ta, tb in zip(a, b):
        if set(ta) != set(tb):
            return {"layers_differ": True, "comparable": False}
        for k in ta:
            x, y = ta[k].float(), tb[k].float()
            if x.shape != y.shape:
                return {"shape_differs": [list(x.shape), list(y.shape)], "comparable": False}
            d = (x - y).abs().max().item()
            r = ((x - y).norm() / x.norm().clamp_min(1e-12)).item()
            max_abs, max_rel = max(max_abs, d), max(max_rel, r)
            n += 1
    return {"comparable": True, "tensors": n, "max_abs": max_abs, "max_rel": max_rel,
            "bitwise": max_abs == 0.0}


def compare(orig: Path, rep: Path, out: Path, sids: List[str]) -> Dict[str, Any]:
    import torch
    A, B = load_rows(orig), load_rows(rep)
    pa = {(e["attack_name"], e["task_id"]): e for e in json.loads((orig / "adv_payloads_cache.json").read_text())} \
        if (orig / "adv_payloads_cache.json").exists() else {}
    pb = {(e["attack_name"], e["task_id"]): e for e in json.loads((rep / "adv_payloads_cache.json").read_text())} \
        if (rep / "adv_payloads_cache.json").exists() else {}
    res, lines = [], []
    for s in sids:
        a, b = A.get(s), B.get(s)
        if a is None or b is None or b.get("error"):
            res.append({"sid": s, "missing": a is None, "rerun_missing_or_error": (b or {}).get("error", b is None)})
            continue
        e = {"sid": s,
             "asr_same": a.get("asr_success") == b.get("asr_success"),
             "ua_same": a.get("utility_success") == b.get("utility_success"),
             "tool_calls_same": a.get("tool_calls") == b.get("tool_calls"),
             "code_hash_same": (a.get("code_version") or {}).get("harness_sha256")
                               == (b.get("code_version") or {}).get("harness_sha256"),
             "attack_hash_same": a.get("attack_sha256") == b.get("attack_sha256"),
             # hardware, dtype, package and capture settings: a fresh directory
             # is never resume-checked, so a difference here must be reported
             "run_settings_diff": {k: [(a.get("run_settings") or {}).get(k), (b.get("run_settings") or {}).get(k)]
                                   for k in set(a.get("run_settings") or {}) | set(b.get("run_settings") or {})
                                   if (a.get("run_settings") or {}).get(k) != (b.get("run_settings") or {}).get(k)}}
        key = (a.get("attack_name"), a.get("task_id"))
        if key in pa or key in pb:
            e["payload_same"] = (pa.get(key) or {}).get("injections") == (pb.get(key) or {}).get("injections")
        ta, tb = _read_tr(orig / f"{s}.transcript.json.gz"), _read_tr(rep / f"{s}.transcript.json.gz")
        xa, xb = _turns_text(ta), _turns_text(tb)
        e["prompts_same"] = xa is not None and xb is not None and [p for p, _ in xa] == [p for p, _ in xb]
        e["completions_same"] = xa is not None and xb is not None and [c for _, c in xa] == [c for _, c in xb]
        if e["prompts_same"] is False and xa and xb:
            e["first_differing_step"] = next((i for i, (u, v) in enumerate(zip(xa, xb)) if u != v), min(len(xa), len(xb)))
        try:
            e["activations"] = _act_diff(torch.load(orig / f"{s}.pt", weights_only=False),
                                         torch.load(rep / f"{s}.pt", weights_only=False))
        except Exception as ex:  # noqa: BLE001
            e["activations"] = {"comparable": False, "error": f"{type(ex).__name__}: {ex}"[:200]}
        res.append(e)

    done = [r for r in res if "asr_same" in r]
    count = lambda k: sum(1 for r in done if r.get(k))
    acts = [r["activations"] for r in done if r["activations"].get("comparable")]
    summary = {
        "episodes": len(sids), "compared": len(done), "not_compared": [r for r in res if "asr_same" not in r],
        "asr_same": count("asr_same"), "ua_same": count("ua_same"), "tool_calls_same": count("tool_calls_same"),
        "prompts_same": count("prompts_same"), "completions_same": count("completions_same"),
        "payload_same": [count("payload_same"), sum(1 for r in done if "payload_same" in r)],
        "code_hash_same": count("code_hash_same"), "attack_hash_same": count("attack_hash_same"),
        "run_settings_differ": sorted({k for r in done for k in r["run_settings_diff"]}),
        "run_settings_diff_example": next((r["run_settings_diff"] for r in done if r["run_settings_diff"]), None),
        "activations_bitwise": sum(1 for x in acts if x["bitwise"]), "activations_compared": len(acts),
        "activations_max_rel": max((x["max_rel"] for x in acts), default=None),
        "activations_max_abs": max((x["max_abs"] for x in acts), default=None),
    }
    lines += [f"# Reproducibility: {orig.name} vs {rep.name}", "",
              f"Episodes: {len(sids)}; compared {len(done)}",
              f"- ASR identical: {summary['asr_same']}/{len(done)}",
              f"- UA identical: {summary['ua_same']}/{len(done)}",
              f"- tool calls identical: {summary['tool_calls_same']}/{len(done)}",
              f"- every step's prompt identical (sha256): {summary['prompts_same']}/{len(done)}",
              f"- every completion identical: {summary['completions_same']}/{len(done)}",
              f"- payload identical: {summary['payload_same'][0]}/{summary['payload_same'][1]}",
              f"- harness code hash identical: {summary['code_hash_same']}/{len(done)}; "
              f"attack code hash identical: {summary['attack_hash_same']}/{len(done)}",
              f"- run settings that differ (GPU type, dtype, packages, capture): "
              f"{summary['run_settings_differ'] or 'none'}"
              + (f" e.g. {summary['run_settings_diff_example']}" if summary['run_settings_differ'] else ""),
              f"- activations bitwise identical: {summary['activations_bitwise']}/{summary['activations_compared']} "
              f"(largest relative difference {summary['activations_max_rel']}, absolute {summary['activations_max_abs']})",
              ""]
    if summary["not_compared"]:
        lines += ["Not compared: " + json.dumps(summary["not_compared"])[:2000], ""]
    diff = [r for r in done if not all(r.get(k) for k in ("asr_same", "ua_same", "completions_same"))]
    if diff:
        lines += ["## Episodes that differ", ""] + [f"- {json.dumps(r)[:600]}" for r in diff] + [""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "numbers.json").write_text(json.dumps({"summary": summary, "episodes": res}, indent=1, default=str))
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("orig")
    c = sub.add_parser("compare")
    c.add_argument("orig")
    c.add_argument("rerun")
    c.add_argument("--samples", required=True, help="the file select wrote")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "select":
        print("\n".join(select(Path(a.orig))))
    else:
        ids = [l.strip() for l in open(a.samples) if l.strip()]
        compare(Path(a.orig), Path(a.rerun), Path(a.out), ids)
