"""
experiments/som_resolver_live_check.py
------------------------------------------
The one thing the offline self-test in environments/som_resolver.py
CANNOT establish: whether our own LIVE rendering produces an observation
the resolver can parse, and whether agent-attack's recorded element
identities actually resolve against it.

Checks three things in order, stopping at the first real failure:
  1. Does obs["text"] from our live env contain bracket-format SoM lines
     at all? (A prior live diagnostic returned an EMPTY text
     obs_nodes_info under observation_type="image_som" -- if the text
     observation is genuinely empty, the resolver has no runtime input
     and the whole identity approach needs a different source.)
  2. Do the recorded identities (victim image hash, click target)
     resolve to ids in OUR observation?
  3. Do our resolved ids differ from the recorded raw ids? (Expected --
     that difference is exactly why the resolver exists. If they happen
     to match, that is fine, just not something to rely on.)

Needs a LIVE Postmill instance and must run INSIDE that job's allocation
(localhost:8080 from the login node reaches the login node, not the
compute node hosting Postmill).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VWA_ADV_REPO = Path(os.environ.get("VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
sys.path.insert(0, str(VWA_ADV_REPO))
os.chdir(str(VWA_ADV_REPO))  # task configs' storage_state paths are relative to here

from browser_env import ScriptBrowserEnv  # noqa: E402

from environments.som_resolver import (  # noqa: E402
    extract_identity,
    load_recorded_observation,
    parse_click_target,
    parse_obs_text,
    resolve_identity,
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--adv_dir", type=str,
                    default="/datasets/nmahomed/vwa-adv/agent-attack/exp_data/agent_adv/reddit_35_upvote_cap")
    ap.add_argument("--exp_data", type=str,
                    default="/datasets/nmahomed/vwa-adv/agent-attack/exp_data")
    ap.add_argument("--login_config_file", type=str, default="config_files/test_reddit/0.json",
                    help="Any real Reddit task config -- used only to trigger storage_state/cookie "
                         "loading before navigating away to obs_url.")
    ap.add_argument("--viewport_height", type=int, default=2048)
    args = ap.parse_args()

    adv_dir = Path(args.adv_dir)
    data = json.loads((adv_dir / "data.json").read_text())
    base = "_".join(adv_dir.name.split("_")[:2])          # reddit_35_upvote_cap -> reddit_35
    recorded = load_recorded_observation(Path(args.exp_data) / "agent_clean" / base)

    victim_id = data.get("victim_som_id")
    target_id = parse_click_target(data.get("target_label"))

    print(f"Task: {adv_dir.name}   (base {base})")
    print(f"  obs_url:            {data.get('obs_url')}")
    print(f"  recorded victim id: {victim_id}")
    print(f"  recorded target:    {data.get('target_label')} -> id {target_id}")

    identities = {}
    for label, som_id in (("victim", victim_id), ("target", target_id)):
        if som_id is None:
            continue
        ident = extract_identity(recorded, som_id)
        identities[label] = (som_id, ident)
        detail = ident.image_hash or ident.text or f"after {ident.anchor_tag} {ident.anchor_text!r} +{ident.anchor_offset}"
        print(f"  identity[{label}]: kind={ident.kind} tag={ident.tag} -> {detail}")

    print(f"\nConstructing ScriptBrowserEnv (headless, image_som, viewport height {args.viewport_height})...")
    env = ScriptBrowserEnv(
        headless=True, observation_type="image_som", current_viewport_only=False,
        viewport_size={"width": 1280, "height": args.viewport_height},
    )

    try:
        print(f"Resetting via {args.login_config_file} (login state only)...")
        env.reset(options={"config_file": args.login_config_file})

        print(f"Navigating to {data['obs_url']} ...")
        env.page.goto(data["obs_url"])
        env.page.wait_for_load_state("load", timeout=15000)

        obs = env._get_obs()

        # --- CHECK 1: is there parseable bracket-format text at all? ---
        text_obs = obs.get("text") if isinstance(obs, dict) else None
        print("\n=== CHECK 1: live text observation ===")
        print(f"obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
        if not isinstance(text_obs, str) or not text_obs.strip():
            print("FAIL: obs['text'] is empty or not a string. The resolver has no runtime input "
                  "from this observation type -- STOP and reconsider the input source before "
                  "wiring anything further.")
            return
        print(f"obs['text'] length: {len(text_obs)} chars")
        print("--- first 800 chars ---")
        print(text_obs[:800])

        Path("/home-mscluster/nmahomed/Honours-Research-IPI-Latent-Analysis/logs/live_obs_text.txt").write_text(text_obs)
        ours = parse_obs_text(text_obs)
        ided = [e for e in ours if e.som_id is not None]
        print(f"\nparsed {len(ours)} lines, {len(ided)} id-bearing elements "
              f"(id range {min((e.som_id for e in ided), default='n/a')}.."
              f"{max((e.som_id for e in ided), default='n/a')})")
        if not ided:
            print("FAIL: text parsed but contained no [id] [TAG] [content] lines -- the live format "
                  "differs from agent-attack's recorded format. Compare the sample above against "
                  "exp_data/agent_clean/<task>/obs_text.txt before proceeding.")
            return

        # --- CHECK 2/3: do recorded identities resolve here, and to what? ---
        print("\n=== CHECK 2/3: identity resolution against OUR observation ===")
        all_ok = True
        for label, (recorded_id, ident) in identities.items():
            our_id = resolve_identity(ours, ident)
            if our_id is None:
                print(f"  {label}: NOT FOUND in our observation (recorded id was {recorded_id}) "
                      f"-- identity kind={ident.kind}")
                all_ok = False
                continue
            match = "same as recorded" if our_id == recorded_id else f"DIFFERS from recorded {recorded_id}"
            print(f"  {label}: resolved to our id {our_id}  ({match})")
            hit = next(e for e in ided if e.som_id == our_id)
            print(f"      our element: [{hit.som_id}] [{hit.tag}] [{hit.content[:120]}]")

        print("\nRESULT:", "identities resolve against live rendering -- safe to wire in."
              if all_ok else
              "at least one identity did NOT resolve -- do not wire in until understood.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
