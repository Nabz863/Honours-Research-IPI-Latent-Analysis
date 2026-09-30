"""
experiments/code_version.py
---------------------------
The code state that produced a row, stamped on every row (audit, 30 Sep).

Two content hashes, not one, because the final runs are split on purpose:
the expensive gradient attacks (GCG, obfuscated activation) run LAST, after
their rebuild, while everything else runs first. Hashing the harness
(environments, experiments, defences, extraction, analysis) separately from
attacks/ makes it checkable that the two runs differ ONLY in attack code.

Computed once per process: Python has already imported the modules, so the
start-time hash is the one that describes what actually ran.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIRS = ("environments", "experiments", "defenses", "extraction", "analysis", "geometry")
ATTACK_DIRS = ("attacks",)


def _hash_dirs(dirs: Iterable[str]):
    h = hashlib.sha256()
    n = 0
    for d in dirs:
        base = ROOT / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            h.update(str(f.relative_to(ROOT)).encode()); h.update(b"\0")
            h.update(f.read_bytes()); h.update(b"\0")
            n += 1
    return h.hexdigest(), n


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:
        return None


@lru_cache(maxsize=1)
def code_version() -> Dict[str, object]:
    harness, n_h = _hash_dirs(HARNESS_DIRS)
    attacks, n_a = _hash_dirs(ATTACK_DIRS)
    diff = _git("diff", "HEAD")
    porcelain = _git("status", "--porcelain") or ""
    return {
        "commit": ((_git("rev-parse", "HEAD") or "").strip() or None),
        "dirty": bool(porcelain.strip()),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "harness_sha256": harness,
        "attacks_sha256": attacks,
        "files": n_h + n_a,
    }
