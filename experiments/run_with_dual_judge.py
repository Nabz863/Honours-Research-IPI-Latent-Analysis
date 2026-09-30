#!/usr/bin/env python3
"""
experiments/run_with_dual_judge.py
----------------------------------
Launcher: installs the dual judge, then runs the curated matrix unchanged.

WHY A LAUNCHER AND NOT A PATCH
    The alternative is editing either the VWA harness or our matrix runner
    to call dual_judge.install(). Both are worse:

    - Editing the harness means the answer to "what did you change in their
      code" grows an entry, and the audit has to justify it. Nothing in
      VisualWebArena is modified by this route; the override is entirely
      additive and lives in our tree.
    - Editing the runner means every future invocation carries the judge
      override whether or not it is wanted, and the two paths (with and
      without) can no longer be compared.

    This way the runner is byte-identical whether the judge override is in
    play, and `python3 experiments/run_with_dual_judge.py ...` versus
    `python3 experiments/run_vwa_adv_curated_matrix.py ...` is the only
    difference.

ORDERING
    install() must run before the harness's evaluators module is imported,
    or at least be able to find it -- see _rebind_aliases in dual_judge for
    why. This launcher installs first, passing --vwa_path so
    evaluation_harness is importable at that point, then hands off. Modules
    the runner imports afterwards pick up the patched functions naturally.

    The install is VERIFIED before handing off. A silently inert patch is
    the failure mode that matters here: the run would complete, the two
    judged tasks would fail exactly as they always have, and no records
    file would exist to make the absence visible. So if verify() reports
    any unpatched reference, this exits non-zero rather than burning GPU
    hours on a run whose judged tasks are still broken.

USAGE
    Every argument except the --dj_* / --vwa_path ones is passed straight
    through to run_vwa_adv_curated_matrix.py:

        python3 experiments/run_with_dual_judge.py \\
            --dj_records analysis_out/judge_verdicts.jsonl \\
            --model_id Qwen/Qwen3-VL-8B-Instruct \\
            --attack_names clean template_ipi comet gcg \\
            --only_tasks reddit_69_modify_comment_cap \\
                         reddit_72_modify_comment_cap \\
            --site reddit --max_steps 15 \\
            --output_dir vwa_adv_qwen3vl_judged

    --dj_no_local          commercial judge only (no 16 GB local model)
    --dj_local_model ID    override the local judge
    --dj_require_openai    exit non-zero if OPENAI_API_KEY is unset, rather
                           than silently falling back to the local judge
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

DEFAULT_VWA = "/datasets/nmahomed/vwa-adv/visualwebarena"
RUNNER = "experiments.run_vwa_adv_curated_matrix"


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--vwa_path", default=os.environ.get("VWA_PATH", DEFAULT_VWA))
    ap.add_argument("--dj_records",
                    default="analysis_out/judge_verdicts.jsonl")
    ap.add_argument("--dj_local_model",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dj_no_local", action="store_true")
    ap.add_argument("--dj_require_openai", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    known, passthrough = ap.parse_known_args()

    if known.help:
        print(__doc__)
        print("\n--- runner's own arguments ---\n")
        sys.argv = ["run_vwa_adv_curated_matrix.py", "--help"]
        runpy.run_module(RUNNER, run_name="__main__")
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if known.dj_require_openai and not key:
        print("FATAL: --dj_require_openai set but OPENAI_API_KEY is empty.\n"
              "  export OPENAI_API_KEY=$(cat ~/.openai_key)",
              file=sys.stderr)
        return 2
    if not key:
        print("WARNING: OPENAI_API_KEY is empty -- the published scoring "
              "path will be SKIPPED and only the local judge will score "
              "the two judged tasks. Those figures will not reproduce the "
              "benchmark's own metric.", file=sys.stderr)

    from environments import dual_judge

    ok = dual_judge.install(
        local_model_id=None if known.dj_no_local else known.dj_local_model,
        record_path=known.dj_records,
        repo_path=known.vwa_path,
    )
    if not ok:
        print("FATAL: dual_judge.install() failed. The two judged tasks "
              "would fail exactly as before and no records file would be "
              "written to show it. Refusing to start the run.\n"
              f"  checked --vwa_path {known.vwa_path}",
              file=sys.stderr)
        return 3

    v = dual_judge.verify()
    if not v["ok"]:
        print(f"FATAL: dual_judge patch is not fully reachable. Still "
              f"holding originals: {v['unpatched']}\n"
              "  These modules would bypass the override silently.",
              file=sys.stderr)
        return 4
    print(f"[launcher] dual judge verified on {len(v['patched'])} "
          f"reference(s); handing off to {RUNNER}", file=sys.stderr)

    sys.argv = ["run_vwa_adv_curated_matrix.py"] + passthrough
    runpy.run_module(RUNNER, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
