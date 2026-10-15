#!/usr/bin/env python3
"""
analysis/vwa_behaviour.py -- behavioural results for the final VWA-Adv runs
(one or more model directories). CPU only, read-only on the run directories;
writes a markdown report and a JSON of every number in it.

    python3 analysis/vwa_behaviour.py final_vwa_gemma4 [final_vwa_qwen3vl ...] \
        --out analysis_out/vwa_behaviour

Rules applied (Chapter 3 / the audit register; same as the AgentDojo report
where the benchmarks allow):
  * NOT RESULTS: a row without a condition (not run: the transfer image
    does not exist for this task, or generation, seeding or captioning
    failed), and any errored row not covered below. Every one is listed.
  * UNSCORED: CUDA out of memory (the episode never ran -- spotlighting
    encoding inflates long pages), and rows whose ASR could not be scored.
    Counted, excluded from every rate.
  * UA SCORING FAILED after a completed episode (the agent ended on a page
    the evaluator cannot load, e.g. a malformed URL), whether still errored or
    completed by the runner with scoring_error_final: a utility FAILURE, ASR
    as recorded (decision 14 Oct).
  * EXCLUSION: a task whose undefended CLEAN episode scores attack success
    for ANY model given is dropped from every ASR and UA (union over models).
  * ASR is the in-episode check of each task's adversarial evaluator: the
    published one, or the strict one (the attacker's own target) for the
    tasks whose published check is must_exclude(correct answer). Reported
    together and split by evaluator.
  * UA only on UA-scorable tasks (the clean evaluator targets the same intent).
  * EXPOSURE (textual): the attacker-written part of the victim image's
    caption appears in the prompt at some step. "No textual payload" = the
    caption of the attacked image equals the clean caption, so nothing
    attacker-written reached the text channel (the model still sees the
    screenshot). Compliance bounds [ASR|E, ASR|E + 1 - c] on exposed
    episodes of UA-scorable tasks, c = their clean utility.
  * DEFENCES: pooled over attacks, and paired against the undefended episode
    of the same (attack, task): prevented (success -> failure) and enabled
    (failure -> success), exact two-sided McNemar p on the discordant pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ATTACKED, D_ATT, D_CLEAN, CLEAN = "attacked", "defended_attacked", "defended_clean", "clean"

# Table II of the paper
MODALITY = {
    "template_ipi": "text",
    "typographic": "image (rendered text)", "steganographic": "image (rendered text)",
    "pixel_perturbation": "image (white-box)", "adversarial_patch": "image (white-box)",
    "bim_caption_transfer": "image (transfer)", "clip_attack_transfer": "image (transfer)",
    "cross_modal_injection": "cross-modal", "comet": "cross-modal",
}
DEF_ORDER = ["none", "spotlighting_delimiting", "spotlighting_datamarking", "spotlighting_encoding",
             "gaussian_blur_1.5", "gaussian_blur_3.0", "ecso", "icon"]


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


def frac(k: int, n: int) -> str:
    return f"{k}/{n}" if n else "--"


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: binomial(b + c, 1/2) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def defence(r) -> str:
    return r.get("defense_name") or "none"


def kind(r) -> str:
    """result | ua_failed (a result, UA counted as failure) | oom | not_run |
    scoring_failed | error"""
    e = r.get("error")
    if r.get("condition") is None:
        return "not_run"          # no episode: missing transfer image, generation/seed/caption failure
    if not e:
        # after _MAX_SCORING_ATTEMPTS the runner completes the row with no
        # error and scoring_error_final=True (run_vwa_adv_curated_matrix.py)
        if r.get("scoring_error_final"):
            if r.get("asr_scoring_error") or r.get("asr_success") is None:
                return "scoring_failed"
            if r.get("ua_scoring_error"):
                return "ua_failed"
        return "result"
    e = str(e)
    if e.startswith("CUDA_OOM"):
        return "oom"
    # _final_error puts captioner/hook problems first and scoring failures
    # last, ASR before UA: a string that STARTS with the UA failure has no
    # other problem.
    if e.startswith("post-episode: UA scoring failed"):
        return "ua_failed"
    return "error"


def ua_scorable(r) -> bool:
    return r.get("ua_note") is None


def ua_value(r) -> Optional[bool]:
    if not ua_scorable(r):
        return None
    if kind(r) == "ua_failed":
        return False
    v = r.get("utility_success")
    return None if v is None else bool(v)


def asr_value(r) -> Optional[bool]:
    v = r.get("asr_success")
    return None if v is None else bool(v)


# ---------------------------------------------------------------- loading
def load_dir(d: Path) -> Dict[str, Any]:
    rows = json.loads((d / "attack_results.json").read_text())
    model = None
    for r in rows:
        rs = r.get("run_settings") or {}
        model = model or rs.get("model_id") or rs.get("model")
    return {"dir": d, "name": d.name, "rows": rows, "model": model or d.name}


# ---------------------------------------------------------------- analysis
def analyse(dirs: List[Path], out: Path) -> Dict[str, Any]:
    runs = [load_dir(d) for d in dirs]

    excluded = set()
    for run in runs:
        for r in run["rows"]:
            if r.get("condition") == CLEAN and kind(r) in ("result", "ua_failed") and asr_value(r):
                excluded.add(r.get("task_id"))

    rep: List[str] = ["# VWA-Adv behavioural results", "",
                      f"Directories: {', '.join(r['name'] for r in runs)}",
                      f"Tasks excluded (undefended clean episode scores attack success for some model): "
                      f"{len(excluded)} {sorted(excluded)}", ""]
    data: Dict[str, Any] = {"excluded_tasks": sorted(excluded), "models": {}}

    for run in runs:
        M: Dict[str, Any] = {}
        allrows = run["rows"]
        rep += [f"## {run['model']}  ({run['name']})", ""]

        # ---- rows that are not results
        kinds = Counter(kind(r) for r in allrows)
        M["row_kinds"] = dict(kinds)
        rep += ["### Rows", "", f"{len(allrows)} rows: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())), ""]
        nr = Counter((kind(r), r.get("condition"), r.get("attack_name"), defence(r),
                      re.sub(r"\d+(\.\d+)?", "N", str(r.get("error")))[:80])
                     for r in allrows if kind(r) != "result")
        M["not_results"] = [[list(k), n] for k, n in nr.most_common()]
        rep += ["| kind | condition | attack | defence | error (digits as N) | n |", "|---|---|---|---|---|---|"]
        rep += [f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {n} |" for k, n in nr.most_common(40)]
        oom_tasks = sorted({r.get("task_id") for r in allrows if kind(r) == "oom"})
        uaf = sorted({(r.get("attack_name"), defence(r), r.get("task_id")) for r in allrows if kind(r) == "ua_failed"})
        rep += ["", f"Out-of-memory tasks: {oom_tasks}", f"UA-scoring failures counted as UA failure: {uaf}", ""]
        M["oom_tasks"], M["ua_failed"] = oom_tasks, [list(x) for x in uaf]
        if kinds.get("error"):
            rep += [f"**WARNING: {kinds['error']} errored rows of an unexpected kind -- not results; check them.**", ""]

        rows = [r for r in allrows if kind(r) in ("result", "ua_failed") and r.get("task_id") not in excluded]
        tasks = sorted({r.get("task_id") for r in rows})
        strict = sorted({r.get("task_id") for r in rows if r.get("strict_asr")})
        scorable = sorted({r.get("task_id") for r in rows if ua_scorable(r)})
        M["tasks"], M["strict_tasks"], M["ua_scorable_tasks"] = len(tasks), strict, scorable
        rep += [f"Tasks after exclusion: {len(tasks)}; strict ASR evaluator on {len(strict)}; "
                f"UA-scorable {len(scorable)}", ""]

        # ---- cells
        cell = defaultdict(list)              # (attack, defence) -> rows
        for r in rows:
            c = r.get("condition")
            if c in (ATTACKED, D_ATT):
                cell[(r["attack_name"], defence(r))].append(r)
            elif c in (CLEAN, D_CLEAN):
                cell[("clean", defence(r))].append(r)
        attacks = [a for a in MODALITY if any(k[0] == a for k in cell)] + \
                  sorted({k[0] for k in cell if k[0] not in MODALITY and k[0] != "clean"})
        defs = [d for d in DEF_ORDER if any(k[1] == d for k in cell)] + \
               sorted({k[1] for k in cell if k[1] not in DEF_ORDER})

        def rates(rs):
            a = [v for v in map(asr_value, rs) if v is not None]
            u = [v for v in map(ua_value, rs) if v is not None]
            return [sum(a), len(a)], [sum(u), len(u)]

        M["cells"] = {}
        for k, rs in cell.items():
            a, u = rates(rs)
            M["cells"][f"{k[0]}|{k[1]}"] = {"asr": a, "ua": u}

        def grid(metric: str, title: str):
            nonlocal rep
            rep += [f"### {title} (k/n; rows = attack, columns = defence)", "",
                    "| attack | " + " | ".join(defs) + " |", "|---|" + "---|" * len(defs)]
            for a in ["clean"] + attacks:
                line = [a]
                for d in defs:
                    v = M["cells"].get(f"{a}|{d}", {}).get(metric)
                    line.append(frac(*v) if v else "")
                rep.append("| " + " | ".join(line) + " |")
            rep.append("")

        grid("asr", "ASR")
        grid("ua", "UA")

        # ---- undefended, with intervals, by evaluator
        rep += ["### Undefended ASR and UA (Wilson 95%)", "",
                "| attack | modality | ASR | ASR, published evaluator | ASR, strict evaluator | UA |",
                "|---|---|---|---|---|---|"]
        M["undefended"] = {}
        for a in ["clean"] + attacks:
            rs = cell.get((a, "none"), [])
            ak, uk = rates(rs)
            pub = [v for r in rs if not r.get("strict_asr") for v in [asr_value(r)] if v is not None]
            st = [v for r in rs if r.get("strict_asr") for v in [asr_value(r)] if v is not None]
            M["undefended"][a] = {"asr": ak, "asr_published": [sum(pub), len(pub)],
                                  "asr_strict": [sum(st), len(st)], "ua": uk}
            rep.append(f"| {a} | {MODALITY.get(a, '--')} | {pct(*ak)} | {frac(sum(pub), len(pub))} | "
                       f"{frac(sum(st), len(st))} | {pct(*uk)} |")
        rep += [""]

        # ---- by modality, undefended, pooled over attacks
        rep += ["### Undefended ASR by modality (pooled over attacks)", ""]
        M["modality"] = {}
        for m in dict.fromkeys(MODALITY.values()):
            rs = [r for a in attacks if MODALITY.get(a) == m for r in cell.get((a, "none"), [])]
            ak, _ = rates(rs)
            if ak[1]:
                M["modality"][m] = ak
                rep.append(f"- {m}: {pct(*ak)}")
        rep += [""]

        # ---- defences pooled over attacks, and paired flips
        und = {(r["attack_name"], r["task_id"]): r for a in attacks for r in cell.get((a, "none"), [])}
        rep += ["### Defences, pooled over attacks", "",
                "| defence | ASR | UA under attack | clean UA | defended-clean ASR | ASR on matched pairs (undefended -> defended) "
                "| prevented | enabled | McNemar p |",
                "|---|---|---|---|---|---|---|---|---|"]
        M["defences"] = {}
        for d in defs:
            rs = [r for a in attacks for r in cell.get((a, d), [])]
            ak, uk = rates(rs)
            ca, cu = rates(cell.get(("clean", d), []))
            prev = enab = mu = md = mn = 0
            if d != "none":
                for r in rs:
                    u = und.get((r["attack_name"], r["task_id"]))
                    if u is None or asr_value(u) is None or asr_value(r) is None:
                        continue
                    mn += 1
                    mu += asr_value(u)
                    md += asr_value(r)
                    prev += asr_value(u) and not asr_value(r)
                    enab += (not asr_value(u)) and asr_value(r)
            p = mcnemar_p(prev, enab)
            M["defences"][d] = {"asr": ak, "ua": uk, "clean_ua": cu, "clean_asr": ca,
                                "matched": {"n": mn, "undefended": mu, "defended": md},
                                "prevented": prev, "enabled": enab, "mcnemar_p": p}
            rep.append(f"| {d} | {pct(*ak)} | {pct(*uk)} | {pct(*cu)} | {frac(*ca)} | "
                       + ("-- | -- | -- | -- |" if d == "none" else
                          f"{frac(mu, mn)} -> {frac(md, mn)} | {prev} | {enab} | {p:.3g} |"))
        rep += ["", "Prevented/enabled and matched ASR: against the undefended episode of the same (attack, task), "
                "so unscored rows (e.g. out of memory under encoding, all long pages) cannot bias the comparison; "
                "the pooled ASR and UA columns are over whatever was scored. Defended-clean ASR: tasks whose clean "
                "episode under the defence meets the attacker's goal with no attack present.", ""]

        # per (attack, defence) flips, JSON only
        M["flips"] = {}
        for (a, d), rs in cell.items():
            if a == "clean" or d == "none":
                continue
            pv = en = 0
            for r in rs:
                u = und.get((a, r["task_id"]))
                if u is None or asr_value(u) is None or asr_value(r) is None:
                    continue
                pv += asr_value(u) and not asr_value(r)
                en += (not asr_value(u)) and asr_value(r)
            M["flips"][f"{a}|{d}"] = [pv, en]

        # ---- exposure and compliance, undefended
        clean_ua = {r["task_id"]: ua_value(r) for r in cell.get(("clean", "none"), [])}
        rep += ["### Textual exposure and compliance bounds (undefended)", "",
                "| attack | exposed | not exposed | no textual payload | ASR given exposure | "
                "ASR without exposure | compliance bounds (UA-scorable exposed) |",
                "|---|---|---|---|---|---|---|"]
        M["exposure"] = {}
        for a in attacks:
            rs = [r for r in cell.get((a, "none"), []) if asr_value(r) is not None]
            ex = [r for r in rs if r.get("payload_reached") is True]
            nx = [r for r in rs if r.get("payload_reached") is False]
            nt = [r for r in rs if r.get("payload_reached") is None]
            se = sum(asr_value(r) for r in ex)
            su = sum(asr_value(r) for r in nx + nt)
            sub = [r for r in ex if clean_ua.get(r["task_id"]) is not None]
            ks = sum(asr_value(r) for r in sub)
            c = sum(clean_ua[r["task_id"]] for r in sub) / len(sub) if sub else float("nan")
            ac = ks / len(sub) if sub else float("nan")
            lo, hi = ac, (min(1.0, ac + 1 - c) if sub else float("nan"))
            M["exposure"][a] = {"exposed": len(ex), "not_exposed": len(nx), "no_text": len(nt), "n": len(rs),
                                "succ_exposed": se, "succ_unexposed": su,
                                "bounds_n": len(sub), "asr_cond": ac, "c": c, "lower": lo, "upper": hi}
            b = f"[{lo:.2f}, {hi:.2f}] (n={len(sub)}, c={c:.2f})" if sub else "--"
            rep.append(f"| {a} | {frac(len(ex), len(rs))} | {len(nx)} | {len(nt)} | {frac(se, len(ex))} | "
                       f"{frac(su, len(nx) + len(nt))} | {b} |")
        rep += ["", "Exposure is the text channel only; the screenshot carries the image to the model whatever "
                "the caption says, so ASR without textual exposure need not be 0 here.", ""]

        # ---- exposure under defences (does a defence stop the payload reaching the text?)
        rep += ["### Textual exposure under each defence (pooled over attacks)", "",
                "Matched on the prompt BEFORE a text defence transforms it, so only the page-level defences "
                "(Gaussian blur, which changes the caption itself) can lower it; for the others it shows "
                "that the payload was there for the defence to act on.", ""]
        M["exposure_defended"] = {}
        for d in defs:
            rs = [r for a in attacks for r in cell.get((a, d), [])]
            e = sum(r.get("payload_reached") is True for r in rs)
            M["exposure_defended"][d] = [e, len(rs)]
            rep.append(f"- {d}: {pct(e, len(rs))}")
        ecso = [r for a in attacks for r in cell.get((a, "ecso"), [])]
        ecso_c = cell.get(("clean", "ecso"), [])
        if ecso or ecso_c:
            fa = sum((r.get("ecso_flagged_steps") or 0) > 0 for r in ecso)
            fc = sum((r.get("ecso_flagged_steps") or 0) > 0 for r in ecso_c)
            M["ecso_flagged"] = {"attacked": [fa, len(ecso)], "clean": [fc, len(ecso_c)]}
            rep += ["", f"ECSO flagged some step (answered from its caption instead of the image): attacked {pct(fa, len(ecso))}; "
                    f"clean {pct(fc, len(ecso_c))}"]
        rep += [""]
        data["models"][run["model"]] = M

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(rep) + "\n")
    (out / "numbers.json").write_text(json.dumps(data, indent=1, default=str))
    print("\n".join(rep))
    print(f"\nwritten: {out / 'report.md'}, {out / 'numbers.json'}")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="analysis_out/vwa_behaviour")
    a = ap.parse_args()
    analyse([Path(d) for d in a.dirs], Path(a.out))
