"""
experiments/code_version.py
---------------------------
The code state and run settings that produced a row, stamped on every row
(audit, 30 Sep; review 3, 30 Sep).

WHAT IS STAMPED
  code_version     harness hash (environments, defenses, extraction, and the
                   experiments modules this process imported), the hash of
                   all of attacks/ (informational), commit, dirty flag.
  attack_sha256    per ATTACK: the attack's own module plus every attacks/
                   module it imports (transitively), plus its constructor
                   settings where the runner passes any. Changing gcg.py
                   changes gcg's and obfuscated_activation's hashes and no
                   one else's.
  run_settings     model id, benchmark version, step budget, and hashes of
                   the evaluator-override directories: things that change
                   what an episode is without changing any code.
  defense_sha256   per DEFENCE where a defence has configuration outside
                   the code (spotlighting's seed, ICON's prober file).

WHY PER-ATTACK
  The final runs are split on purpose: the expensive gradient attacks (GCG,
  obfuscated activation) run LAST, after their rebuild, into the same
  output directory. A whole-directory attacks hash would make the second
  run refuse (or, with an override, stop checking anything), and would
  make a later retry of a first-run attack refuse too (review 3, #13).

A per-file manifest of the harness is written to <output_dir>/code_versions/
so a refused resume can say WHICH files differ, and ALLOW_CHANGED_FILES can
permit exactly those files and nothing else.

Computed once per process: Python has already imported the modules, so the
start-time hash is the one that describes what actually ran.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
# Code that can change what an episode does. experiments/ is NOT hashed
# wholesale -- it also holds unrelated scripts, and editing one of those
# must not block a resume -- only the experiments modules actually imported
# by this process. analysis/ and geometry/ never run inside an episode.
HARNESS_DIRS = ("environments", "defenses", "extraction")
ATTACK_DIRS = ("attacks",)


def _files_in(dirs: Iterable[str]):
    out = []
    for d in dirs:
        base = ROOT / d
        if base.is_dir():
            out.extend(sorted(base.rglob("*.py")))
    return out


def _imported_experiments():
    import sys as _sys
    out = []
    for m in list(_sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "experiments" and rel.suffix == ".py":
            out.append(ROOT / rel)
    return sorted(set(out))


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hash_files(files):
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode()); h.update(b"\0")
        h.update(f.read_bytes()); h.update(b"\0")
    return h.hexdigest(), len(files)


def _hash_dirs(dirs: Iterable[str]):
    return _hash_files(_files_in(dirs))


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:
        return None


def _harness_files():
    return sorted(set(_files_in(HARNESS_DIRS) + _imported_experiments()))


@lru_cache(maxsize=1)
def harness_manifest() -> Dict[str, str]:
    """relative path -> sha256 of every harness file. Written once per
    output directory, never onto rows (it is large)."""
    return {str(f.relative_to(ROOT)): _sha_bytes(f.read_bytes()) for f in _harness_files()}


@lru_cache(maxsize=1)
def code_version() -> Dict[str, object]:
    harness_files = _harness_files()
    harness, n_h = _hash_files(harness_files)
    attacks, n_a = _hash_dirs(ATTACK_DIRS)
    # Scoped to the code directories: output directories live inside the
    # repo, so an unscoped `git status` is always dirty (review C18).
    code = list(HARNESS_DIRS) + list(ATTACK_DIRS) + ["experiments"]
    diff = _git("diff", "HEAD", "--", *code)
    porcelain = _git("status", "--porcelain", "--untracked-files=all", "--", *code) or ""
    porcelain = "\n".join(l for l in porcelain.splitlines()
                          if l.strip().endswith(".py"))
    import logging
    logging.getLogger(__name__).info(
        "code_version harness files: %s",
        [str(f.relative_to(ROOT)) for f in harness_files])
    return {
        "commit": ((_git("rev-parse", "HEAD") or "").strip() or None),
        "dirty": bool(porcelain.strip()),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "harness_sha256": harness,
        "attacks_sha256": attacks,
        "files": n_h + n_a,
    }


# ---------------------------------------------------------------------------
# Per-attack hashes
# ---------------------------------------------------------------------------
def _attack_imports(path: Path) -> List[Path]:
    """attacks/ modules imported by `path` (anywhere in the file, including
    inside functions): `from .x import`, `from attacks.x import`,
    `import attacks.x`."""
    out = []
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return out
    base = ROOT / "attacks"
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 1:
                names.append(mod)
                # `from . import x`
                if not mod:
                    names.extend(a.name for a in node.names)
            elif mod == "attacks":
                names.extend(a.name for a in node.names)
            elif mod.startswith("attacks."):
                names.append(mod[len("attacks."):])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("attacks."):
                    names.append(a.name[len("attacks."):])
        for n in names:
            if not n:
                continue
            p = base / (n.split(".")[0] + ".py")
            if p.exists():
                out.append(p)
    return out


def _attack_closure(path: Path) -> List[Path]:
    seen, todo = set(), [path]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo.extend(_attack_imports(p))
    return sorted(seen)


def attack_code_sha(obj_or_cls: Any, settings: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Hash of one attack: its module, every attacks/ module that module
    imports (transitively), and `settings` (constructor arguments the
    runner chose, e.g. obfuscated_activation's target layer and budget)."""
    import sys as _sys
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    mod = _sys.modules.get(cls.__module__)
    f = getattr(mod, "__file__", None)
    if not f:
        return None
    files = _attack_closure(Path(f).resolve())
    h = hashlib.sha256()
    for p in files:
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            rel = p.name
        h.update(rel.encode()); h.update(b"\0"); h.update(p.read_bytes()); h.update(b"\0")
    h.update(cls.__qualname__.encode())
    if settings:
        h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Run settings and defence configuration
# ---------------------------------------------------------------------------
def dir_sha(path: Optional[str], pattern: str = "*") -> Optional[str]:
    """Content hash of every file matching `pattern` under a directory, or
    None when unset. Use pattern="*.py" for code directories, so compiled
    caches do not count."""
    if not path:
        return None
    p = Path(path)
    if not p.is_dir():
        return f"missing:{p.name}"      # no absolute path in the hash
    h = hashlib.sha256()
    for f in sorted(p.rglob(pattern)):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            h.update(str(f.relative_to(p)).encode()); h.update(b"\0")
            h.update(f.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def files_sha(paths, base=None) -> str:
    """Content hash of a list of files (missing ones hash as missing).
    Names are hashed RELATIVE to `base`, so moving the data, or a node on
    which the path resolves differently, does not change the hash
    (review 5, M6)."""
    h = hashlib.sha256()
    items = []
    for x in paths:
        fp = Path(x)
        name = str(fp)
        if base is not None:
            try:
                name = str(fp.resolve().relative_to(Path(base).resolve()))
            except ValueError:
                name = fp.name
        items.append((name, fp))
    for name, fp in sorted(items, key=lambda t: t[0]):
        h.update(name.encode()); h.update(b"\0")
        h.update(fp.read_bytes() if fp.is_file() else b"<missing>"); h.update(b"\0")
    return h.hexdigest()


def package_sha(name: str) -> Optional[str]:
    """Content hash of an installed package's files (code AND data -- an
    editable install with local changes is caught, which a version string
    is not)."""
    try:
        import importlib
        mod = importlib.import_module(name)
        return dir_sha(str(Path(mod.__file__).parent), "*")
    except Exception:
        return None


def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def file_sha(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    return _sha_bytes(Path(path).read_bytes())


def spotlight_seed_sha() -> str:
    """Hash (never the value) of the spotlighting seed in force."""
    from defenses.spotlighting import spotlight_seed
    return _sha_bytes(spotlight_seed())[:16]


def model_identity(model) -> Optional[str]:
    for attr in ("name_or_path",):
        v = getattr(model, attr, None)
        if v:
            return str(v)
    cfg = getattr(model, "config", None)
    v = getattr(cfg, "_name_or_path", None)
    return str(v) if v else None


def write_manifest(output_dir: Path) -> Path:
    """<output_dir>/code_versions/<harness sha>.json -- lets a later resume
    name the files that changed."""
    cv = code_version()
    d = Path(output_dir) / "code_versions"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{cv['harness_sha256']}.json"
    if not p.exists():
        from experiments.episode_utils import atomic_write_json
        atomic_write_json(p, {"code_version": cv, "files": harness_manifest()}, indent=1)
    return p


def _changed_files(output_dir: Optional[Path], old_sha: Optional[str]) -> Optional[List[str]]:
    if output_dir is None or not old_sha:
        return None
    p = Path(output_dir) / "code_versions" / f"{old_sha}.json"
    if not p.exists():
        return None
    try:
        old = json.loads(p.read_text()).get("files", {})
    except Exception:
        return None
    new = harness_manifest()
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


# ---------------------------------------------------------------------------
# Resume check
# ---------------------------------------------------------------------------
class MixedCodeError(SystemExit):
    pass


def check_resume_compatible(rows, current: Dict[str, object], attack_names=None, *,
                            attack_shas: Optional[Dict[str, Optional[str]]] = None,
                            run_settings: Optional[Dict[str, Any]] = None,
                            defense_shas: Optional[Dict[str, Optional[str]]] = None,
                            output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Refuses to resume into a directory whose completed rows were written
    under different conditions (review C9; review 3 #4, #13). For every
    completed row:
      * the harness hash must match;
      * run_settings must match (model, benchmark version, step budget,
        evaluator-override directories);
      * for rows of an attack being run now, that attack's own hash must
        match (attack_shas); other attacks' rows are not affected by it;
      * for rows of a defence with external configuration (defense_shas),
        that configuration must match.

    Narrow override for a harness change that provably cannot affect the
    completed rows: ALLOW_CHANGED_FILES=path1,path2 permits a harness
    mismatch only when the differing files are exactly within that list
    (read from the manifest in <output_dir>/code_versions/). Returns a
    record of the override to stamp on new rows, else None.
    Blanket override: ALLOW_MIXED_CODE=1 (logged; avoid)."""
    import logging
    log = logging.getLogger(__name__)
    attack_names = set(attack_names or [])
    attack_shas = attack_shas or {}
    defense_shas = defense_shas or {}
    bad = []
    harness_mismatch_shas = set()
    for r in rows:
        if r.get("error") is not None:
            continue
        cv = r.get("code_version") or {}
        sid = r.get("sample_id")
        if cv.get("harness_sha256") != current["harness_sha256"]:
            harness_mismatch_shas.add(cv.get("harness_sha256"))
        if run_settings is not None and (r.get("run_settings") or {}) != run_settings:
            bad.append((sid, "run_settings", r.get("run_settings")))
            continue
        an = r.get("attack_name")
        if an in attack_names and an not in ("clean", None):
            if an in attack_shas and attack_shas[an] is not None:
                if r.get("attack_sha256") != attack_shas[an]:
                    bad.append((sid, f"attack code ({an})", r.get("attack_sha256")))
                    continue
            elif cv.get("attacks_sha256") != current["attacks_sha256"]:
                bad.append((sid, "attacks", cv.get("attacks_sha256")))
                continue
        dn = r.get("defense_name")
        if dn in defense_shas and defense_shas[dn] is not None:
            if r.get("defense_sha256") != defense_shas[dn]:
                bad.append((sid, f"defence config ({dn})", r.get("defense_sha256")))

    override = None
    if harness_mismatch_shas:
        allowed = {s.strip() for s in os.environ.get("ALLOW_CHANGED_FILES", "").split(",") if s.strip()}
        changed_all = set()
        unknown = False
        for old in harness_mismatch_shas:
            ch = _changed_files(output_dir, old)
            if ch is None:
                unknown = True
            else:
                changed_all.update(ch)
        if allowed and not unknown and changed_all and changed_all <= allowed:
            override = {"allowed_changed_files": sorted(changed_all),
                        "previous_harness_sha256": sorted(s for s in harness_mismatch_shas if s)}
            log.warning("ALLOW_CHANGED_FILES: resuming across a harness change limited to %s",
                        sorted(changed_all))
        else:
            detail = ("changed files: " + ", ".join(sorted(changed_all))) if changed_all else \
                     "no manifest for the earlier code, so the changed files cannot be listed"
            if allowed and changed_all and not changed_all <= allowed:
                detail += f" (ALLOW_CHANGED_FILES does not cover: {sorted(changed_all - allowed)})"
            bad.insert(0, ("*", "harness", detail))

    if not bad:
        return override
    msg = (f"{len(bad)} completed-row problem(s) (first: {bad[0]}). Resuming would mix "
           "code states or run settings in one directory. Use a fresh --output_dir; or, "
           "for a harness change that cannot affect the completed rows, set "
           "ALLOW_CHANGED_FILES to exactly the changed files.")
    if os.environ.get("ALLOW_MIXED_CODE") == "1":
        log.warning("ALLOW_MIXED_CODE=1: %s", msg)
        return {"allow_mixed_code": True, "problems": [str(b)[:200] for b in bad[:20]]}
    raise MixedCodeError("FATAL: " + msg)
