"""
experiments/vwa_adv_stepwise_diagnostic.py
-----------------------------------------------
Before building any real stepwise-attack integration against
agent-attack's data: confirm our own SoM labeling actually lines up
with theirs. data.json's victim_som_id/position were recorded against
the ORIGINAL paper's SoM implementation in mid-2024 -- if this
project's browser_env version numbers page elements differently
(a real, live possibility across library versions, not a hypothetical),
that recorded ground truth doesn't transfer, and any attack built on
top of it would silently target the wrong element. This script only
answers that one question -- it does not attempt an attack yet.

Does NOT replay previous_action -- navigates directly to obs_url instead
(UNVERIFIED ASSUMPTION, stated explicitly: that this reproduces a
comparable state to what replaying previous_action would have. obs_url
looks like a plain subreddit listing page, so this is plausible, but
genuinely unconfirmed until this runs for real).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VWA_ADV_REPO = Path(os.environ.get("VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
sys.path.insert(0, str(VWA_ADV_REPO))
os.chdir(str(VWA_ADV_REPO))  # storage_state paths in task configs are relative to here

from browser_env import ScriptBrowserEnv


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, required=True,
                         help="e.g. /datasets/nmahomed/vwa-adv/agent-attack/exp_data/agent_adv/reddit_35_upvote_cap/data.json")
    parser.add_argument("--login_config_file", type=str, default="config_files/test_reddit/0.json",
                         help="Any real Reddit task config -- used purely to trigger env.reset()'s "
                              "storage_state/cookie loading before navigating away to obs_url.")
    args = parser.parse_args()

    with open(args.data_json) as f:
        data = json.load(f)

    print(f"Task data loaded from {args.data_json}")
    print(f"  intent: {data.get('intent')}")
    print(f"  obs_url: {data.get('obs_url')}")
    print(f"  previous_action (NOT replayed, see module docstring): {data.get('previous_action')}")
    print(f"  RECORDED victim_som_id: {data.get('victim_som_id')}")
    print(f"  RECORDED position: {data.get('position')}")
    print(f"  target_label: {data.get('target_label')}")
    print(f"  adversarial_goal: {data.get('adversarial_goal')}")

    print("\nConstructing ScriptBrowserEnv (headless, image_som)...")
    env = ScriptBrowserEnv(
        headless=True, observation_type="image_som",
        current_viewport_only=False, viewport_size={"width": 1280, "height": 2048},
    )

    try:
        print(f"\nResetting via {args.login_config_file} (for login state only)...")
        obs, info = env.reset(options={"config_file": args.login_config_file})

        print(f"Navigating directly to obs_url: {data['obs_url']}")
        env.page.goto(data["obs_url"])
        env.page.wait_for_load_state("load", timeout=10000)

        obs = env._get_obs()
        metadata = env.observation_handler.get_observation_metadata()

        print("\n=== Our own SoM labeling at this URL ===")
        print(f"Metadata type: {type(metadata)}")
        print(f"Raw metadata (first 2000 chars): {str(metadata)[:2000]}")

        recorded_id = data.get("victim_som_id")
        print(f"\n>>> Does SoM id {recorded_id} exist in our own labeling above, and does it look like "
              f"the right element (an image, near position {data.get('position')})? <<<")
        print(">>> Compare visually too -- screenshot saved below. <<<")

        screenshot_path = "/tmp/vwa_adv_stepwise_diagnostic.png"
        env.page.screenshot(path=screenshot_path)
        print(f"\nSaved raw screenshot to {screenshot_path} -- pull it locally to look at directly, "
              f"e.g.: scp mscluster-login:{screenshot_path} .")

    finally:
        env.close()


if __name__ == "__main__":
    main()
