"""
experiments/cross_modal_delivery_check.py
----------------------------------------------
Proves -- with evidence, not source-reading -- that BOTH VWA-Adv
injection channels actually reach a live observation:

  adv_url2caption : rewrites an image's alt text (the accessibility-tree
                    "description" the agent reads).
  adv_url2image   : replaces the rendered pixels via
                    node.src = 'data:image/png;base64,...'.

Why this needs testing rather than assuming: with captioning_fn=None
(our setup), processors.py takes a DIFFERENT branch than the captioned
one. In that branch the caption path calls urljoin() to absolutise a
relative src before lookup, but the IMAGE path does a bare
`if image_url in adv_url2image` with NO urljoin. If Postmill serves
relative src attributes, visual injection silently no-ops while text
injection works -- the worst kind of failure, since nothing errors.

Also recovers the image URLs themselves. They don't appear in our live
obs['text'] (that annotation only happens when a captioner populates
url2caption), so this reads them straight off the DOM via Playwright --
which is also exactly the form adv_url2*'s keys must take.

Needs a LIVE Postmill instance, run INSIDE that job's allocation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VWA_ADV_REPO = Path(os.environ.get("VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
sys.path.insert(0, str(VWA_ADV_REPO))
os.chdir(str(VWA_ADV_REPO))

from PIL import Image  # noqa: E402
from browser_env import ScriptBrowserEnv  # noqa: E402

MARKER = "ZZINJECTEDCAPTIONMARKERZZ"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--adv_dir", type=str,
                    default="/datasets/nmahomed/vwa-adv/agent-attack/exp_data/agent_adv/reddit_35_upvote_cap")
    ap.add_argument("--login_config_file", type=str, default="config_files/test_reddit/0.json")
    ap.add_argument("--viewport_height", type=int, default=2048)
    args = ap.parse_args()

    data = json.loads((Path(args.adv_dir) / "data.json").read_text())
    victim_hash = None
    # The recorded victim image's content hash, from agent-attack's own data.
    base = "_".join(Path(args.adv_dir).name.split("_")[:2])
    recorded = Path("/datasets/nmahomed/vwa-adv/agent-attack/exp_data/agent_clean") / base / "obs_text.txt"
    if recorded.exists():
        import re
        for line in recorded.read_text().splitlines():
            if line.startswith(f"[{data.get('victim_som_id')}] "):
                m = re.search(r"/([0-9a-f]{32,64})\.(?:jpg|jpeg|png|gif|webp)", line, re.IGNORECASE)
                victim_hash = m.group(1) if m else None
                break
    print(f"Task: {Path(args.adv_dir).name}")
    print(f"  recorded victim_som_id: {data.get('victim_som_id')}")
    print(f"  recorded victim hash:   {victim_hash}")

    env = ScriptBrowserEnv(
        headless=True, observation_type="image_som", current_viewport_only=False,
        viewport_size={"width": 1280, "height": args.viewport_height},
    )

    try:
        env.reset(options={"config_file": args.login_config_file})
        env.page.goto(data["obs_url"])
        env.page.wait_for_load_state("load", timeout=15000)

        # --- STEP 1: what image URLs does the live DOM actually expose? ---
        print("\n=== STEP 1: live image src attributes (these are adv_url2* keys) ===")
        srcs = env.page.eval_on_selector_all("img", "els => els.map(e => e.getAttribute('src'))")
        for s in srcs[:10]:
            relative = s is not None and not s.startswith(("http://", "https://", "www."))
            print(f"  {'RELATIVE!' if relative else 'absolute '} {s}")
        if any(s and not s.startswith(("http://", "https://", "www.")) for s in srcs):
            print("  >>> WARNING: relative src values present. adv_url2image does NOT urljoin in the")
            print("  >>> no-captioner branch, so keys must match the RAW src exactly, not the")
            print("  >>> absolutised URL. Build keys from these strings verbatim.")

        target_src = None
        if victim_hash:
            target_src = next((s for s in srcs if s and victim_hash in s), None)
        if target_src is None:
            target_src = next((s for s in srcs if s and "submission_thumbnail" in s), None)
            print(f"\n  victim hash not matched; falling back to first thumbnail: {target_src}")
        else:
            print(f"\n  matched victim image src: {target_src}")
        if target_src is None:
            print("FAIL: no usable image src found on this page.")
            return

        # --- STEP 2: does adv_url2caption actually reach obs['text']? ---
        print("\n=== STEP 2: caption injection ===")
        obs_clean = env._get_obs()
        obs_adv = env._get_obs(adv_url2caption={target_src: MARKER}, adv_url2image=None)
        landed = MARKER in obs_adv.get("text", "")
        print(f"  marker present in adversarial obs['text']: {landed}")
        if landed:
            for line in obs_adv["text"].splitlines():
                if MARKER in line:
                    print(f"    -> {line[:200]}")
                    break
        else:
            print("  FAIL: caption injection did not land. Key format is the likely cause --")
            print("        compare the src strings above against what was passed as the key.")

        # --- STEP 3: does adv_url2image actually replace the pixels? ---
        print("\n=== STEP 3: image injection ===")
        red = Image.new("RGB", (200, 200), color=(255, 0, 0))
        env._get_obs(adv_url2caption=None, adv_url2image={target_src: red})
        srcs_after = env.page.eval_on_selector_all("img", "els => els.map(e => e.getAttribute('src'))")
        replaced = [s for s in srcs_after if s and s.startswith("data:image/png;base64,")]
        print(f"  imgs now carrying an injected data: URL: {len(replaced)}")
        if replaced:
            print("  PASS: pixel replacement reached the live DOM (the screenshot the model sees).")
        else:
            print("  FAIL: no data: URL present -- adv_url2image did not match. Most likely the")
            print("        missing-urljoin issue: pass the RAW src attribute string as the key.")

        print("\nSUMMARY: caption channel =", "OK" if landed else "BROKEN",
              "| image channel =", "OK" if replaced else "BROKEN")
        print("Both must be OK for cross_modal_injection to be measuring what it claims.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
