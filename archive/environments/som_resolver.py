"""
environments/som_resolver.py
------------------------------
Resolves agent-attack's recorded SoM ids (data.json's victim_som_id and
target_label "click [N]") to STABLE ELEMENT IDENTITIES, then back to
whatever SoM id our own live rendering assigns at runtime.

WHY THIS EXISTS: SoM ids are not stable. Confirmed empirically on a real
run -- the SAME page, SAME session, differing ONLY in viewport height
(720 vs 2048), assigned id 38 to a right-sidebar text label in one case
and a left-column upvote arrow in the other. Their recorded ids come
from a mid-2024 collection with unknown viewport/scroll settings, so
trusting a recorded id number directly would silently target the wrong
element -- no error, no warning, wrong data.

THE FIX: match on element IDENTITY, which is stable across renderings:
  - IMG elements: the image URL's content hash (Postmill serves
    thumbnails at .../submission_thumbnail_1x/<sha256>.jpg -- the hash
    is content-derived, so it survives any layout/viewport change).
  - Elements with distinctive text (e.g. [A] "60 comments"): the text.
  - Elements with EMPTY text (the upvote [BUTTON]s, which is most of the
    click-type targets): anchored relative to the nearest preceding
    element that DOES have distinctive text, by offset in id order.
    E.g. reddit_35's target [41] is "the 1st id-bearing element after
    the [A] link reading '60 comments'".

STATUS: parsing + identity extraction + re-resolution are all verifiable
OFFLINE against agent-attack's own recorded obs_text.txt files (see
self-test at the bottom of this file -- no Postmill, no browser, no GPU
needed). What is NOT yet verified is re-resolution against OUR OWN live
observation text, which needs a running Postmill instance; the offline
self-test deliberately resolves recorded-against-recorded as a sanity
floor, which catches parsing/anchoring bugs but cannot catch a genuine
recorded-vs-live mismatch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Matches lines like:
#   [41] [BUTTON] []
#   [40] [A] [60 comments]
#   [] [StaticText] [1109]
#   [38] [IMG] [, description: a group of people..., url: http://.../aa2b1c.jpg]
_LINE_RE = re.compile(r"^\[([^\]]*)\]\s+\[([^\]]*)\]\s+\[(.*)\]\s*$")
_URL_RE = re.compile(r"url:\s*(\S+)")
# Postmill thumbnail filenames are sha256 content hashes -- stable across
# any layout change, unlike ids or coordinates.
_HASH_RE = re.compile(r"/([0-9a-f]{32,64})\.(?:jpg|jpeg|png|gif|webp)", re.IGNORECASE)


@dataclass
class SoMElement:
    som_id: Optional[int]      # None for StaticText (no interactable id)
    tag: str                   # A, BUTTON, IMG, INPUT, StaticText, ...
    content: str               # raw bracket contents
    line_index: int            # position in the original file

    @property
    def url(self) -> Optional[str]:
        m = _URL_RE.search(self.content)
        return m.group(1) if m else None

    @property
    def image_hash(self) -> Optional[str]:
        u = self.url
        if not u:
            return None
        m = _HASH_RE.search(u)
        return m.group(1) if m else None

    @property
    def text(self) -> str:
        """Human-visible text, with IMG's description/url metadata stripped
        (that metadata is caption text, not element identity, and the
        caption is exactly what an attack MODIFIES -- so it must never be
        used as an identity anchor)."""
        if self.tag.upper() == "IMG":
            return ""
        return self.content.strip()


@dataclass
class ElementIdentity:
    """A viewport/scroll-independent description of one element."""
    kind: str                          # "image_hash" | "text" | "anchored"
    tag: str
    image_hash: Optional[str] = None
    text: Optional[str] = None
    anchor_text: Optional[str] = None  # for "anchored": preceding distinctive text
    anchor_tag: Optional[str] = None
    anchor_offset: int = 0             # steps forward in id-order from the anchor
    debug: Dict[str, Any] = field(default_factory=dict)


def parse_obs_text(text: str) -> List[SoMElement]:
    """Parses an obs_text.txt (ours or theirs) into SoMElements."""
    elements: List[SoMElement] = []
    for i, line in enumerate(text.splitlines()):
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        raw_id, tag, content = m.group(1).strip(), m.group(2).strip(), m.group(3)
        som_id = int(raw_id) if raw_id.isdigit() else None
        elements.append(SoMElement(som_id=som_id, tag=tag, content=content, line_index=i))
    return elements


def _id_bearing(elements: List[SoMElement]) -> List[SoMElement]:
    return [e for e in elements if e.som_id is not None]


def extract_identity(elements: List[SoMElement], som_id: int) -> ElementIdentity:
    """
    Builds a stable identity for the element carrying `som_id` in this
    observation. Raises if that id isn't present at all -- better to fail
    loudly than resolve to something plausible-but-wrong.
    """
    ided = _id_bearing(elements)
    by_id = {e.som_id: e for e in ided}
    if som_id not in by_id:
        raise KeyError(
            f"SoM id {som_id} not present in this observation "
            f"(ids present range {min(by_id) if by_id else 'n/a'}..{max(by_id) if by_id else 'n/a'}). "
            "The recorded observation and this one disagree -- do not guess."
        )
    target = by_id[som_id]

    # 1. Best case: an image with a content-hash URL.
    h = target.image_hash
    if h:
        return ElementIdentity(kind="image_hash", tag=target.tag, image_hash=h,
                               debug={"url": target.url})

    # 2. Distinctive own text, provided it's unique in this observation.
    t = target.text
    if t:
        same = [e for e in ided if e.tag == target.tag and e.text == t]
        if len(same) == 1:
            return ElementIdentity(kind="text", tag=target.tag, text=t)

    # 3. Empty or ambiguous text (the upvote BUTTONs): anchor to the
    #    nearest preceding id-bearing element with unique, non-empty text.
    pos = ided.index(target)
    for back in range(pos - 1, -1, -1):
        cand = ided[back]
        ct = cand.text
        if not ct:
            continue
        same = [e for e in ided if e.tag == cand.tag and e.text == ct]
        if len(same) != 1:
            continue
        return ElementIdentity(
            kind="anchored", tag=target.tag,
            anchor_text=ct, anchor_tag=cand.tag, anchor_offset=pos - back,
            debug={"anchor_som_id": cand.som_id, "target_som_id": som_id},
        )

    raise ValueError(
        f"Could not build a stable identity for SoM id {som_id} ({target.tag!r}, text={t!r}) -- "
        "no content hash, no unique text, and no unique preceding text anchor."
    )


def resolve_identity(elements: List[SoMElement], identity: ElementIdentity) -> Optional[int]:
    """
    Finds `identity` in a (possibly differently-rendered) observation and
    returns ITS SoM id there. Returns None when not found -- callers must
    treat that as "skip this task", never as a default id.
    """
    ided = _id_bearing(elements)

    if identity.kind == "image_hash":
        for e in ided:
            if e.image_hash == identity.image_hash:
                return e.som_id
        return None

    if identity.kind == "text":
        hits = [e for e in ided if e.tag == identity.tag and e.text == identity.text]
        return hits[0].som_id if len(hits) == 1 else None

    if identity.kind == "anchored":
        hits = [i for i, e in enumerate(ided)
                if e.tag == identity.anchor_tag and e.text == identity.anchor_text]
        if len(hits) != 1:
            return None
        idx = hits[0] + identity.anchor_offset
        if idx >= len(ided):
            return None
        cand = ided[idx]
        # Tag must still match, or we've landed on a structurally different
        # element and should refuse rather than return a wrong id.
        return cand.som_id if cand.tag == identity.tag else None

    return None


def parse_click_target(target_label: str) -> Optional[int]:
    """'click [41]' -> 41. Returns None for non-click targets (string /
    list / GPT4EVAL criteria), which need no SoM resolution at all."""
    if not isinstance(target_label, str):
        return None
    m = re.match(r"\s*click\s*\[(\d+)\]\s*$", target_label)
    return int(m.group(1)) if m else None


def load_recorded_observation(clean_task_dir: Path) -> List[SoMElement]:
    """Loads agent-attack's own recorded obs_text.txt for a base task
    (e.g. .../exp_data/agent_clean/reddit_35/)."""
    p = Path(clean_task_dir) / "obs_text.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- needed as the identity reference.")
    return parse_obs_text(p.read_text())


# ---------------------------------------------------------------------------
# Offline self-test -- no Postmill, no browser, no GPU.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, glob, os

    ap = argparse.ArgumentParser(description="Offline check of SoM identity resolution.")
    ap.add_argument("--exp_data", type=str,
                    default="/datasets/nmahomed/vwa-adv/agent-attack/exp_data")
    args = ap.parse_args()

    adv_dirs = sorted(glob.glob(os.path.join(args.exp_data, "agent_adv", "reddit_*/")))
    ok = skipped = failed = 0

    for d in adv_dirs:
        name = os.path.basename(d.rstrip("/"))
        dj = Path(d) / "data.json"
        if not dj.exists():
            print(f"SKIP  {name}: no data.json")
            skipped += 1
            continue
        data = json.loads(dj.read_text())

        base = "_".join(name.split("_")[:2])  # reddit_35_upvote_cap_1 -> reddit_35
        clean_dir = Path(args.exp_data) / "agent_clean" / base
        try:
            recorded = load_recorded_observation(clean_dir)
        except FileNotFoundError as e:
            print(f"SKIP  {name}: {e}")
            skipped += 1
            continue

        report = []
        for label, som_id in (("victim", data.get("victim_som_id")),
                              ("target", parse_click_target(data.get("target_label")))):
            if som_id is None:
                report.append(f"{label}=n/a")
                continue
            try:
                ident = extract_identity(recorded, som_id)
                back = resolve_identity(recorded, ident)
                mark = "OK" if back == som_id else f"MISMATCH(got {back})"
                report.append(f"{label}[{som_id}]:{ident.kind}:{mark}")
                if back != som_id:
                    failed += 1
                else:
                    ok += 1
            except (KeyError, ValueError) as e:
                report.append(f"{label}[{som_id}]:ERROR({type(e).__name__})")
                failed += 1

        print(f"{name}: " + "  ".join(report))

    print(f"\nresolved OK: {ok}   failed: {failed}   skipped dirs: {skipped}")
    print("NOTE: this is recorded-against-recorded -- a sanity floor for parsing/anchoring. "
          "It cannot detect a genuine recorded-vs-live mismatch; that needs a live Postmill run.")
