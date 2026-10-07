#!/usr/bin/env python3
"""
analysis/agentdojo_behaviour.py -- behavioural results for the final AgentDojo
runs (one or more model directories). CPU only, read-only on the run
directories; writes a markdown report and a JSON of every number in it.

    python3 analysis/agentdojo_behaviour.py final_agentdojo_gemma4 [final_agentdojo_qwen3vl ...] \
        --out analysis_out/agentdojo_behaviour

Rules applied, each stated in Chapter 3 / the audit register:
  * errored rows are not results; a row with error=None but a None score
    (scoring_errors, P2a.2) is unscored for that score only
  * EXCLUSION: a pair whose undefended CLEAN episode scores attack success for
    ANY model given is dropped from every ASR and UA (union over models, so
    all models are compared on one set)
  * clean and defended-clean utility once per (suite, user task): those
    episodes do not depend on the injection task (dedup pointers carry the
    representative's scores)
  * EXPOSURE: the payload reached the model if its text appears in the
    episode's transcript -- matched on lower-case letters and digits only,
    because AgentDojo renders tool results as YAML (quoting, escaping and line
    wrapping) and VWA strips quotes
  * conditional ASR = successes among exposed episodes; compliance bounds
    [ASR|E, ASR|E + 1 - c] with c = clean utility on the same user tasks
    (Frechet bounds; the point estimate ASR|E / c assumes independence)
  * Attention Tracker detection at ANY threshold from the recorded focus
    scores (an episode is flagged at tau if any checked tool result scores
    <= tau); behaviour at 0.3 comes from the band re-run, not from here
  * ICON: pairs it fixed (attack success -> failure) and broke (failure ->
    success) against the undefended episode of the same pair
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ATTACKED = "attacked"
D_ATT = "defended_attacked"
D_CLEAN = "defended_clean"


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "--"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}) [{100 * lo:.1f}, {100 * hi:.1f}]"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def pair_key(r) -> Tuple[str, str, str]:
    return (r.get("suite_name"), r.get("user_task_id"), r.get("injection_task_id"))


def user_key(r) -> Tuple[str, str]:
    return (r.get("suite_name"), r.get("user_task_id"))


def transcript_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        tr = json.loads(gzip.decompress(path.read_bytes()))
    except Exception:
        return None
    out = []
    for m in tr.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    out.append(str(b.get("text") or b.get("content") or ""))
    return norm(" ".join(out))


def auc(pos: List[float], neg: List[float]) -> float:
    """P(score_neg > score_pos): lower focus = more suspicious."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if q > p else (0.5 if q == p else 0.0)
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------- loading
def load_dir(d: Path) -> Dict[str, Any]:
    rows = json.loads((d / "attack_results.json").read_text())
    pay = {}
    p = d / "adv_payloads_cache.json"
    if p.exists():
        for e in json.loads(p.read_text()):
            pay[(e["attack_name"], e["task_id"])] = e
    model = None
    for r in rows:
        rs = r.get("run_settings") or {}
        model = model or rs.get("model_id") or rs.get("model")
    return {"dir": d, "name": d.name, "rows": rows, "payloads": pay, "model": model or d.name}


def injection_defaults() -> Dict[str, Dict[str, str]]:
    """Per suite, each injection vector's default text (to strip it from an
    appended payload). Needs the agentdojo package, as the runs did."""
    from environments.agentdojo_tasks import load_tasks
    import os
    _, ctx = load_tasks(benchmark_version=os.environ.get("AGENTDOJO_BENCHMARK_VERSION", "v1.1.2"))
    return {name: s.get_injection_vector_defaults() for name, s in ctx["suites"].items()}


def payload_markers(entry: Dict[str, Any], suite: str, defaults) -> List[str]:
    """The attacker-written part of each injected vector, as a normalised
    40-character marker."""
    out = []
    replace = bool((entry.get("metadata") or {}).get("replaces_vector_content"))
    for vec, text in (entry.get("injections") or {}).items():
        base = (defaults.get(suite) or {}).get(vec, "")
        part = text if replace or not text.startswith(base) else text[len(base):]
        m = norm(part)[:40]
        if len(m) >= 12:
            out.append(m)
    return sorted(set(out))


# ---------------------------------------------------------------- analysis
def analyse(dirs: List[Path], out: Path) -> None:
    runs = [load_dir(d) for d in dirs]
    try:
        defaults = injection_defaults()
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: no agentdojo defaults ({type(e).__name__}: {e}); exposure not computed", file=sys.stderr)
        defaults = None

    # exclusion: union over models of pairs whose undefended clean episode "succeeds"
    excluded = set()
    for run in runs:
        for r in run["rows"]:
            if r.get("condition") == "clean" and not r.get("error") and r.get("asr_success") is True:
                excluded.add(pair_key(r))

    report: List[str] = ["# AgentDojo behavioural results", ""]
    report += [f"Directories: {', '.join(r['name'] for r in runs)}",
               f"Pairs excluded (clean episode scores attack success for some model): {len(excluded)}"
               + (f" -- {sorted(excluded)[:12]}" if excluded else ""), ""]
    data: Dict[str, Any] = {"excluded_pairs": sorted(map(list, excluded)), "models": {}}

    for run in runs:
        rows = [r for r in run["rows"] if pair_key(r) not in excluded]
        M = {"cells": {}, "errors": {}, "exposure": {}, "compliance": {}, "at": {}, "icon": {},
             "per_suite": {}, "template": {}}
        report += [f"## {run['model']}  ({run['name']})", ""]

        # errors and unscored
        err = Counter((r.get("condition"), r.get("attack_name"), r.get("defense_name"),
                       re.sub(r"\d+", "N", str(r.get("error")))[:70]) for r in rows if r.get("error"))
        unscored = Counter((r.get("condition"), r.get("attack_name"), r.get("defense_name"),
                            tuple(sorted((r.get("scoring_errors") or {}).keys())))
                           for r in rows if not r.get("error") and r.get("scoring_errors"))
        M["errors"] = [[list(k), n] for k, n in err.most_common()]
        report += ["### Rows that are not results", "",
                   "| condition | attack | defence | error (digits as N) | n |", "|---|---|---|---|---|"]
        report += [f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {n} |" for k, n in err.most_common(15)]
        if unscored:
            report += ["", "Unscored scores (benchmark check raised; excluded from that rate only): "
                       + "; ".join(f"{k[:3]} {list(k[3])}: {n}" for k, n in unscored.items())]
        report += [""]

        # clean utility, once per user task
        clean_by_user: Dict[Tuple, Optional[bool]] = {}
        for r in rows:
            if r.get("condition") == "clean" and not r.get("error"):
                clean_by_user.setdefault(user_key(r), r.get("utility_success"))
        cu = [v for v in clean_by_user.values() if v is not None]
        M["clean_utility"] = {"k": sum(cu), "n": len(cu)}
        # The clean episode runs once per PAIR (same input for every
        # injection task of a user task); greedy decoding should make those
        # runs agree. Disagreement = nondeterminism, reported, not hidden.
        per_user = defaultdict(set)
        for r in rows:
            if r.get("condition") == "clean" and not r.get("error") and r.get("utility_success") is not None:
                per_user[user_key(r)].add(bool(r["utility_success"]))
        disagree = sorted(u for u, v in per_user.items() if len(v) > 1)
        pp = [r["utility_success"] for r in rows if r.get("condition") == "clean" and not r.get("error")
              and r.get("utility_success") is not None]
        M["clean_utility_per_pair"] = {"k": sum(pp), "n": len(pp)}
        M["clean_disagreeing_user_tasks"] = [list(u) for u in disagree]

        # main table
        cells = defaultdict(list)
        for r in rows:
            if r.get("error"):
                continue
            cond = r.get("condition")
            if cond in (ATTACKED, D_ATT):
                cells[(r.get("attack_name"), r.get("defense_name") or "none")].append(r)
        dclean = defaultdict(dict)
        for r in rows:
            if r.get("condition") == D_CLEAN and not r.get("error"):
                dclean[r.get("defense_name")].setdefault(user_key(r), r.get("utility_success"))

        report += ["### ASR and UA (Wilson 95% intervals)", "",
                   f"Clean utility (one episode per user task): {pct(sum(cu), len(cu))}; "
                   f"per pair: {pct(sum(pp), len(pp))}; user tasks whose clean runs disagree "
                   f"across pairs: {len(disagree)}" + (f" {disagree[:8]}" if disagree else ""), "",
                   "| attack | defence | ASR | UA under attack | defended-clean UA |", "|---|---|---|---|---|"]
        attacks = sorted({a for a, _ in cells})
        defs = ["none"] + sorted({d for _, d in cells if d != "none"})
        for a in attacks:
            for d in defs:
                rs = cells.get((a, d))
                if not rs:
                    continue
                asr = [r["asr_success"] for r in rs if r.get("asr_success") is not None]
                ua = [r["utility_success"] for r in rs if r.get("utility_success") is not None]
                dc = [v for v in dclean.get(d, {}).values() if v is not None] if d != "none" else cu
                M["cells"][f"{a}|{d}"] = {"asr": [sum(asr), len(asr)], "ua": [sum(ua), len(ua)],
                                          "dclean_ua": [sum(dc), len(dc)]}
                report.append(f"| {a} | {d} | {pct(sum(asr), len(asr))} | {pct(sum(ua), len(ua))} | "
                              f"{pct(sum(dc), len(dc))} |")
        report += [""]

        # per suite, undefended
        report += ["### Undefended ASR by suite", "", "| attack | " + " | ".join(
            s for s in ("workspace", "travel", "banking", "slack")) + " |", "|---|---|---|---|---|"]
        for a in attacks:
            rs = cells.get((a, "none"), [])
            line = [a]
            for s in ("workspace", "travel", "banking", "slack"):
                v = [r["asr_success"] for r in rs if r.get("suite_name") == s and r.get("asr_success") is not None]
                M["per_suite"][f"{a}|{s}"] = [sum(v), len(v)]
                line.append(f"{sum(v)}/{len(v)}")
            report.append("| " + " | ".join(line) + " |")
        report += [""]

        # template breakdown
        tp = defaultdict(lambda: [0, 0])
        for r in cells.get(("template_ipi", "none"), []):
            if r.get("asr_success") is None:
                continue
            i = r.get("attack_meta_template_idx")
            if i is None:   # a reused payload: its metadata is in the cache
                i = ((run["payloads"].get(("template_ipi", r.get("task_id"))) or {}).get("metadata") or {}).get("template_idx")
            if i is None:
                continue
            tp[i][0] += bool(r["asr_success"])
            tp[i][1] += 1
        if tp:
            report += ["### template_ipi by template (pinned per pair)", ""]
            report += [f"- template {i}: {pct(k, n)}" for i, (k, n) in sorted(tp.items(), key=lambda x: str(x[0]))]
            M["template"] = {str(i): v for i, v in tp.items()}
            report += [""]

        # exposure and compliance, undefended
        if defaults is not None:
            report += ["### Exposure and compliance bounds (undefended)", "",
                       "| attack | exposed | ASR (blind) | ASR given exposure | compliance bounds | "
                       "point est. (indep.) |", "|---|---|---|---|---|---|"]
            for a in attacks:
                rs = [r for r in cells.get((a, "none"), []) if r.get("asr_success") is not None]
                exp_n = succ_e = n_tr = 0
                users_e = set()
                for r in rs:
                    entry = run["payloads"].get((a, r.get("task_id")))
                    if not entry:
                        continue
                    marks = payload_markers(entry, r.get("suite_name"), defaults)
                    txt = transcript_text(run["dir"] / f"{r['sample_id']}.transcript.json.gz")
                    if txt is None or not marks:
                        continue
                    n_tr += 1
                    if any(mk in txt for mk in marks):
                        exp_n += 1
                        succ_e += bool(r["asr_success"])
                        users_e.add(user_key(r))
                blind = sum(bool(r["asr_success"]) for r in rs)
                # c on the user tasks the exposed episodes come from
                cu_e = [clean_by_user[u] for u in users_e if clean_by_user.get(u) is not None]
                c = sum(cu_e) / len(cu_e) if cu_e else float("nan")
                ac = succ_e / exp_n if exp_n else float("nan")
                lo, hi = ac, min(1.0, ac + 1 - c) if exp_n else float("nan")
                pe = ac / c if exp_n and c else float("nan")
                M["exposure"][a] = {"exposed": exp_n, "checked": n_tr, "succ_exposed": succ_e, "blind": [blind, len(rs)]}
                M["compliance"][a] = {"asr_cond": ac, "c": c, "lower": lo, "upper": hi, "point_indep": pe}
                report.append(f"| {a} | {pct(exp_n, n_tr)} | {pct(blind, len(rs))} | {pct(succ_e, exp_n)} | "
                              f"[{lo:.3f}, {hi:.3f}] (c={c:.3f}) | {pe:.3f} |")
            report += ["", "c = clean utility on the user tasks of the exposed episodes. Bounds need no "
                       "independence assumption; the point estimate does.", ""]

        # Attention Tracker detection at any threshold
        at_by_attack = defaultdict(list)
        for r in rows:
            if (r.get("defense_name") == "attention_tracker" and r.get("condition") == D_ATT
                    and not r.get("error") and r.get("defense_scores")):
                at_by_attack[r.get("attack_name")].append(min(r["defense_scores"]))
        # clean: the representative episodes only (dedup rows copy its scores)
        at_cl = [min(r["defense_scores"]) for r in rows if r.get("defense_name") == "attention_tracker"
                 and r.get("condition") == D_CLEAN and not r.get("error") and r.get("defense_scores")
                 and not r.get("deduplicated_to")]
        if at_by_attack or at_cl:
            atk = sorted(at_by_attack)
            report += ["### Attention Tracker: episode-level detection at any threshold", "",
                       "An episode is flagged at tau if any checked tool result scores <= tau "
                       "(from the recorded focus scores; behaviour at 0.3 comes from the band re-run).", "",
                       "| tau | clean flagged (FPR) | " + " | ".join(f"{a} flagged" for a in atk) + " |",
                       "|---|---|" + "---|" * len(atk)]
            for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                kc = sum(s <= tau for s in at_cl)
                M["at"][str(tau)] = {"clean": [kc, len(at_cl)]}
                cells_ = []
                for a in atk:
                    ka = sum(s <= tau for s in at_by_attack[a])
                    M["at"][str(tau)][a] = [ka, len(at_by_attack[a])]
                    cells_.append(pct(ka, len(at_by_attack[a])))
                report.append(f"| {tau} | {pct(kc, len(at_cl))} | " + " | ".join(cells_) + " |")
            report += [""]
            for a in atk:
                a_ = auc(at_by_attack[a], at_cl)
                M["at"][f"auc|{a}"] = a_
                report.append(f"- AUC, {a} vs clean (min episode focus score): {a_:.3f}")
            report += [""]

        # ICON flips
        for a in attacks:
            und = {pair_key(r): r for r in cells.get((a, "none"), []) if r.get("asr_success") is not None}
            ic = {pair_key(r): r for r in cells.get((a, "icon"), []) if r.get("asr_success") is not None}
            both = set(und) & set(ic)
            if not both:
                continue
            fixed = sum(1 for p in both if und[p]["asr_success"] and not ic[p]["asr_success"])
            broke = sum(1 for p in both if not und[p]["asr_success"] and ic[p]["asr_success"])
            same_calls = sum(1 for p in both if und[p].get("tool_calls") == ic[p].get("tool_calls"))
            M["icon"][a] = {"pairs": len(both), "fixed": fixed, "broke": broke, "identical_tool_calls": same_calls}
            report.append(f"- ICON (oracle rectifier), {a}: {len(both)} pairs; fixed {fixed}, broke {broke}; "
                          f"identical tool calls {same_calls}/{len(both)}")
        report += [""]
        data["models"][run["model"]] = M

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(report) + "\n")
    (out / "numbers.json").write_text(json.dumps(data, indent=1, default=str))
    print("\n".join(report))
    print(f"\nwritten: {out / 'report.md'}, {out / 'numbers.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="analysis_out/agentdojo_behaviour")
    a = ap.parse_args()
    analyse([Path(d) for d in a.dirs], Path(a.out))
