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
  * Attention Tracker detection at other thresholds from the recorded focus
    scores: exact at tau >= the run's threshold (0.5); below it, bounds,
    because scores after the first withheld result come from a trajectory a
    run at tau might not take. Exact values and behaviour at 0.3 come from
    the band re-run
  * a lower-threshold Attention Tracker condition (attention_tracker_tNN) is
    read from the band merge (analysis/at_threshold_band.py merge), passed as
    --at_rows DIR=FILE: it REPLACES the directory's own tNN rows, which are
    only the re-run band episodes. Its detection is then exact, and is checked
    against the bounds from the 0.5 run
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


# A backslash escape as YAML (double-quoted style) and Python repr write it:
# \n, \t, \", \\, \xNN, \uNNNN, \UNNNNNNNN, and the line-fold "\<newline>".
# AgentDojo renders a tool result whose string starts or ends with whitespace,
# or holds non-ASCII text, in double-quoted style -- every important_instructions
# payload -- so a newline in the payload is the two characters "\n" in the
# transcript, and norm() alone would keep the "n".
_ESC = re.compile(r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|[\s\S])")


def norm_rendered(s: str) -> str:
    """norm() of text as the model saw it in a rendered tool result.
    Escapes can be stacked: the runner wraps every tool result in
    json.dumps({"result": ...}) (environments/agentdojo_llm.py), so a result
    AgentDojo already rendered with escapes (YAML double-quoted, or the
    Python repr of a dict -- every travel tool) reaches the model as "\\\\n".
    Collapse doubled backslashes until stable, then drop each escape."""
    s = s or ""
    while "\\\\" in s:
        s = s.replace("\\\\", "\\")
    return norm(_ESC.sub(" ", s))


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
    return norm_rendered(" ".join(out))


def auc(pos: List[float], neg: List[float]) -> float:
    """P(score_neg > score_pos): lower focus = more suspicious."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if q > p else (0.5 if q == p else 0.0)
    return wins / (len(pos) * len(neg))


AT_RUN_TAU = 0.5      # threshold the Attention Tracker runs used (released default)


def at_flag_bounds(scores, tau: float, run_tau: float = AT_RUN_TAU) -> Tuple[bool, bool]:
    """(flagged for certain, possibly flagged) at threshold tau, given an
    episode's focus scores in the order the tool results arrived in a run at
    run_tau. Scores after the first one <= run_tau come from a trajectory that
    a run at another threshold might not have taken."""
    first = next((s for s in scores if s <= run_tau), None)
    if first is None:              # never withheld: every score is valid
        f = any(s <= tau for s in scores)
        return f, f
    if tau >= run_tau or first <= tau:
        return True, True          # withheld at or before the same point
    return False, True             # first flag in (tau, run_tau]: undetermined


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
def apply_at_rows(run: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Replace the run's lower-threshold Attention Tracker rows with the full
    merged set (re-run band episodes + carried rows)."""
    merged = json.loads(Path(path).read_text())
    names = {r.get("defense_name") for r in merged}
    if not merged or not all(n and n.startswith("attention_tracker_t") for n in names):
        raise SystemExit(f"FATAL: {path} is not an at_threshold_band merge (defences {sorted(map(str, names))})")
    # the merge must cover every 0.5 row: at_threshold_band.py leaves out a
    # row still errored at the lower threshold (and exits 1), so a partial
    # file would silently shrink the condition
    n_hi = sum(1 for r in run["rows"] if r.get("defense_name") == "attention_tracker")
    for n in names:
        n_lo = sum(1 for r in merged if r.get("defense_name") == n)
        if n_lo != n_hi:
            raise SystemExit(f"FATAL: {path}: {n_lo} {n} rows for {n_hi} attention_tracker rows -- the band "
                             f"merge is incomplete (an episode still errored at the lower threshold); "
                             f"resubmit the band job first")
    kept = [r for r in run["rows"] if r.get("defense_name") not in names]
    run["rows"] = kept + merged
    run["at_lo"] = {n: len([r for r in merged if r.get("defense_name") == n]) for n in names}
    return run


def lo_tau(name: str) -> float:
    return int(name.rsplit("_t", 1)[1]) / 100


def analyse(dirs: List[Path], out: Path, at_rows: Optional[Dict[str, Path]] = None) -> None:
    runs = [load_dir(d) for d in dirs]
    for run in runs:
        if at_rows and run["name"] in at_rows:
            apply_at_rows(run, at_rows[run["name"]])
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
                       "point est. (indep.) | successes NOT exposed (should be 0) | exposed by suite |",
                       "|---|---|---|---|---|---|---|---|"]
            for a in attacks:
                rs = [r for r in cells.get((a, "none"), []) if r.get("asr_success") is not None]
                exp_n = succ_e = n_tr = succ_unexp = 0
                cu_e: List[bool] = []
                by_suite = defaultdict(lambda: [0, 0])
                for r in rs:
                    entry = run["payloads"].get((a, r.get("task_id")))
                    if not entry:
                        continue
                    marks = payload_markers(entry, r.get("suite_name"), defaults)
                    txt = transcript_text(run["dir"] / f"{r['sample_id']}.transcript.json.gz")
                    if txt is None or not marks:
                        continue
                    n_tr += 1
                    by_suite[r.get("suite_name")][1] += 1
                    if any(mk in txt for mk in marks):
                        exp_n += 1
                        by_suite[r.get("suite_name")][0] += 1
                        succ_e += bool(r["asr_success"])
                        if clean_by_user.get(user_key(r)) is not None:
                            cu_e.append(bool(clean_by_user[user_key(r)]))
                    else:
                        succ_unexp += bool(r["asr_success"])
                blind = sum(bool(r["asr_success"]) for r in rs)
                # c = clean utility of the exposed EPISODES (weighted by episode,
                # as the bound is over episodes)
                c = sum(cu_e) / len(cu_e) if cu_e else float("nan")
                ac = succ_e / exp_n if exp_n else float("nan")
                lo, hi = ac, (min(1.0, ac + 1 - c) if exp_n else float("nan"))
                pe = ac / c if exp_n and c else float("nan")
                M["exposure"][a] = {"exposed": exp_n, "checked": n_tr, "succ_exposed": succ_e,
                                    "succ_unexposed": succ_unexp, "blind": [blind, len(rs)],
                                    "by_suite": {k: v for k, v in by_suite.items()}}
                M["compliance"][a] = {"asr_cond": ac, "c": c, "lower": lo, "upper": hi, "point_indep": pe}
                report.append(f"| {a} | {pct(exp_n, n_tr)} | {pct(blind, len(rs))} | {pct(succ_e, exp_n)} | "
                              f"[{lo:.3f}, {hi:.3f}] (c={c:.3f}) | {pe:.3f} | {succ_unexp} | "
                              + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_suite.items())) + " |")
            report += ["", "c = clean utility of the exposed episodes' user tasks, per episode. A success with no "
                       "exposure means the payload match missed (or the agent met the goal unprompted). Bounds need no "
                       "independence assumption; the point estimate does.", ""]

        # Attention Tracker detection at other thresholds, from the run at
        # AT_RUN_TAU. Withholding a result changes the rest of the episode, so
        # only the scores up to the FIRST one <= AT_RUN_TAU are valid at another
        # threshold. Exact at tau >= AT_RUN_TAU; below it, an episode whose
        # first flagged score lies in (tau, AT_RUN_TAU] is undetermined (the band
        # re-run resolves it), so detection is reported as [lower, upper].
        at_att = defaultdict(list)
        for r in rows:
            if (r.get("defense_name") == "attention_tracker" and r.get("condition") == D_ATT
                    and not r.get("error") and r.get("defense_scores")):
                at_att[r.get("attack_name")].append(r["defense_scores"])
        # clean: the representative episodes only (dedup rows copy its scores)
        at_cl = [r["defense_scores"] for r in rows if r.get("defense_name") == "attention_tracker"
                 and r.get("condition") == D_CLEAN and not r.get("error") and r.get("defense_scores")
                 and not r.get("deduplicated_to")]
        if at_att or at_cl:
            atk = sorted(at_att)
            report += [f"### Attention Tracker: episode-level detection at other thresholds (run at {AT_RUN_TAU})", "",
                       "Flagged = some tool result withheld. Exact at tau >= the run threshold; below it, "
                       "[lower, upper]: episodes whose first flagged score lies in (tau, run threshold] are "
                       "undetermined until the band re-run.", "",
                       "| tau | clean flagged (FPR) | " + " | ".join(f"{a} flagged" for a in atk) + " |",
                       "|---|---|" + "---|" * len(atk)]

            def cell(eps, tau):
                lo = hi = 0
                for sc in eps:
                    a, b = at_flag_bounds(sc, tau)
                    lo += a
                    hi += b
                n = len(eps)
                txt = pct(lo, n) if lo == hi else f"[{100 * lo / n:.1f}, {100 * hi / n:.1f}]% ({lo}-{hi}/{n})"
                return {"lo": [lo, n], "hi": [hi, n]}, txt

            for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                c, ct = cell(at_cl, tau)
                M["at"][str(tau)] = {"clean": c}
                cells_ = []
                for a in atk:
                    v, t = cell(at_att[a], tau)
                    M["at"][str(tau)][a] = v
                    cells_.append(t)
                report.append(f"| {tau} | {ct} | " + " | ".join(cells_) + " |")
            report += [""]

        # Lower-threshold condition(s) from the band merge: detection exact.
        # Each must fall inside the bounds the 0.5 run gave at that tau.
        M["at_exact"] = {}
        for name in sorted((run.get("at_lo") or {})):
            tau = lo_tau(name)
            flagged = lambda r: (r.get("defense_detections") or 0) > 0
            att = defaultdict(list)
            for r in rows:
                if r.get("defense_name") == name and r.get("condition") == D_ATT and not r.get("error"):
                    att[r.get("attack_name")].append(flagged(r))
            cl = [flagged(r) for r in rows if r.get("defense_name") == name and r.get("condition") == D_CLEAN
                  and not r.get("error") and not r.get("deduplicated_to")]
            # Per-episode check against the 0.5 run: an episode the 0.5 scores
            # say is certainly flagged (or certainly not) at tau must be so here.
            hi_by_sid = {r["sample_id"]: r for r in rows if r.get("defense_name") == "attention_tracker"}
            hi_name = lambda sid: sid.replace(f"_{name}_", "_attention_tracker_", 1)
            check = defaultdict(lambda: [0, 0])          # key -> [compared, violations]
            for r in rows:
                if r.get("defense_name") != name or r.get("error") or r.get("deduplicated_to"):
                    continue
                h = hi_by_sid.get(r.get("carried_from") or hi_name(r["sample_id"]))
                if not h or h.get("error") or not h.get("defense_scores"):
                    continue
                k = "clean" if r.get("condition") == D_CLEAN else r.get("attack_name")
                certain, possible = at_flag_bounds(h["defense_scores"], tau)
                check[k][0] += 1
                check[k][1] += (certain and not flagged(r)) or (flagged(r) and not possible)
            report += [f"### Attention Tracker at {tau} (band re-run: exact)", "",
                       f"{run['at_lo'][name]} rows from the merge.", "",
                       "| episodes | flagged | episodes checked against the 0.5 bounds | outside them (must be 0) |",
                       "|---|---|---|---|"]
            ex = {"clean": [sum(cl), len(cl)]}
            for a in sorted(att):
                ex[a] = [sum(att[a]), len(att[a])]
            M["at_exact"][name] = {"tau": tau, "flagged": ex, "checked": {}, "violations": {}}
            for k, (f, n) in ex.items():
                c, v = check.get(k, [0, 0])
                M["at_exact"][name]["checked"][k], M["at_exact"][name]["violations"][k] = c, v
                report.append(f"| {k} | {pct(f, n)} | {c} | {v}{'  <-- CHECK' if v else ''} |")
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
    ap.add_argument("--at_rows", nargs="*", default=[], metavar="DIR=FILE",
                    help="band merge for a run directory, e.g. "
                         "final_agentdojo_gemma4=analysis_out/at_t30_rows_gemma4.json")
    a = ap.parse_args()
    at = {}
    for spec in a.at_rows:
        d, _, f = spec.partition("=")
        if not f:
            ap.error(f"--at_rows wants DIR=FILE, got {spec!r}")
        at[Path(d).name] = Path(f)
    analyse([Path(d) for d in a.dirs], Path(a.out), at)
