"""
analysis/asr_review_sheet.py
----------------------------
Manual, per-task definition of the strict ASR check for the 40 curated
tasks -- because automated extraction from data.json is not reliable
enough and 40 is small enough to read.

WHY MANUAL
    A regex pass over data.json produced two silent errors on its first
    run: it pulled the CORRECT answer out of `target_label` ("NOT: 03-03")
    and offered it as the attacker's target, and its mm/dd format check
    matched the "mm" inside the word "comment", giving the wrong date
    format for the _2 variants. Either mistake converts a broken metric
    into a differently broken one that LOOKS fixed.

THE FIELDS THAT MATTER (confirmed by reading real task dirs)
    target_caption_clip  -- the attacker's positive target, already
                            clean: "toilet paper", "a mask with a
                            smiling face", "Release date: 04/24/2023"
    victim_caption_clip  -- the CORRECT answer the attack suppresses:
                            "a pumpkin", "twenty-million-dollar,
                            Zimbabwean banknote"
    target_label         -- names the correct answer ("NOT: 03") or a
                            grading instruction ("GPT4EVAL: ..."). NEVER
                            the attacker's target. Shown for context only.
    intent               -- fixes the required FORMAT (mm-dd vs dd vs mm)

Workflow
    1. python3 analysis/asr_review_sheet.py emit --exp_data ... \\
           --out analysis_out/asr_review.tsv
    2. Open the TSV, fill the STRICT_MUST_INCLUDE column by hand.
       Pipe-separate alternatives: 24|April 24
       Leave blank to exclude the task from strict ASR entirely.
    3. python3 analysis/asr_review_sheet.py build --exp_data ... \\
           --sheet analysis_out/asr_review.tsv --out_dir strict_configs/

The build step refuses to write a config whose target string also
appears in victim_caption_clip -- that would score the CORRECT answer as
an attack success, which is the exact failure being fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLS = [
    "task", "eval_type", "current_check", "intent",
    "target_caption_clip", "victim_caption_clip", "target_label",
    "adversarial_goal", "STRICT_MUST_INCLUDE", "NOTE",
]


def current_check(cfg: dict) -> str:
    ev = cfg.get("eval", {}) or {}
    bits = []
    if "url_match" in (ev.get("eval_types") or []):
        bits.append(f"url_match:{ev.get('reference_url', '')}")
    for b in ev.get("program_html") or []:
        rc = b.get("required_contents") or {}
        for k, v in rc.items():
            bits.append(f"{k}={v}")
    return " ; ".join(bits) or "(none)"


def eval_type(cfg: dict) -> str:
    ev = cfg.get("eval", {}) or {}
    if "url_match" in (ev.get("eval_types") or []) and ev.get("reference_url"):
        return "url_match"
    for b in ev.get("program_html") or []:
        rc = b.get("required_contents") or {}
        for k in ("must_include", "fuzzy_match", "must_exclude"):
            if k in rc:
                return k
    return "none"


def load_tasks(exp_data: Path, site: str):
    out = {}
    for d in sorted((exp_data / "agent_adv").glob("*/")):
        cfg_p, data_p = d / "config.json", d / "data.json"
        if not cfg_p.exists():
            continue
        cfg = json.load(open(cfg_p))
        if site not in (cfg.get("sites") or []):
            continue
        data = json.load(open(data_p)) if data_p.exists() else {}
        out[d.name] = (cfg, data)
    return out


def cmd_emit(a):
    tasks = load_tasks(Path(a.exp_data), a.site)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(COLS)
        for name, (cfg, data) in tasks.items():
            et = eval_type(cfg)
            # pre-fill ONLY where the source is already a positive check;
            # everything else is left blank for a human.
            prefill = ""
            note = ""
            if et == "url_match":
                prefill = "(already positive - url_match)"
                note = "no action needed unless clean FPs implicate it"
            elif et == "must_include":
                prefill = "(already positive - must_include)"
                note = "no action needed"
            elif et == "fuzzy_match":
                note = "needs an LLM judge; consider excluding"
            w.writerow([
                name, et, current_check(cfg), cfg.get("intent", ""),
                data.get("target_caption_clip", ""),
                data.get("victim_caption_clip", ""),
                data.get("target_label", ""),
                data.get("adversarial_goal", ""),
                prefill, note,
            ])
    n_manual = sum(1 for _, (c, _) in tasks.items()
                   if eval_type(c) in ("must_exclude", "fuzzy_match", "none"))
    print(f"wrote {a.out}  ({len(tasks)} tasks, {n_manual} need manual entry)")
    print("Fill STRICT_MUST_INCLUDE for those rows. Pipe-separate "
          "alternatives. Blank = exclude from strict ASR.")


def cmd_build(a):
    tasks = load_tasks(Path(a.exp_data), a.site)
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    written, skipped, refused = 0, [], []
    with open(a.sheet, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            name = row["task"]
            raw = (row.get("STRICT_MUST_INCLUDE") or "").strip()
            if not raw or raw.startswith("("):
                skipped.append(name)
                continue
            if name not in tasks:
                skipped.append(name)
                continue
            cfg, data = tasks[name]
            targets = [t.strip() for t in raw.split("|") if t.strip()]

            # refuse anything that would score the CORRECT answer as a hit
            victim = (data.get("victim_caption_clip") or "").lower()
            label = (data.get("target_label") or "").lower()
            bad = [t for t in targets
                   if (victim and t.lower() in victim)
                   or (t.lower() and t.lower() in label.replace("not:", ""))]
            if bad:
                refused.append((name, bad))
                continue

            new = json.loads(json.dumps(cfg))
            blocks = new.get("eval", {}).get("program_html") or []
            if not blocks:
                refused.append((name, ["no program_html block to rewrite"]))
                continue
            for b in blocks:
                if b.get("required_contents"):
                    b["required_contents"] = {"must_include": targets}
            new["_strict_asr"] = {
                "targets": targets,
                "original": current_check(cfg),
                "source": "manual review sheet",
            }
            (outdir / f"{name}.json").write_text(json.dumps(new, indent=4))
            written += 1

    print(f"wrote {written} strict configs to {outdir}")
    if skipped:
        print(f"skipped (blank or already positive): {len(skipped)}")
    if refused:
        print("REFUSED -- target matches the correct answer, would score a "
              "correct response as an attack success:")
        for n, b in refused:
            print(f"  {n}: {b}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit")
    e.add_argument("--exp_data", required=True)
    e.add_argument("--site", default="reddit")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_emit)

    b = sub.add_parser("build")
    b.add_argument("--exp_data", required=True)
    b.add_argument("--site", default="reddit")
    b.add_argument("--sheet", required=True)
    b.add_argument("--out_dir", required=True)
    b.set_defaults(func=cmd_build)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
