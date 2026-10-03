"""
tests/test_vwa_adv_tasks.py
------------------------------
Tests for environments/vwa_adv_tasks.py.

load_tasks() is tested against REAL generated config_files/test_reddit/
data (see this project's own confirmed finding that generate_test_data.py
needs no live server, just env vars set). _early_stop() is tested
against controlled synthetic trajectories -- pure list/dict logic, no
browser needed, ported directly from the real run.py's own early_stop()
(fetched in full, not partially).

run_vwa_adv_episode() itself is NOT tested here -- it needs a live
Postmill instance and a real Playwright browser, neither available in
this environment. See vwa_adv_tasks.py's own STATUS note.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# visualwebarena on the path, as the runner does (VWA_ADV_REPO); skipped,
# not failed, when no checkout is present.
_VWA_ROOT = (os.environ.get("VWA_ADV_REPO") or os.environ.get("VWA_ROOT")
             or "/datasets/nmahomed/vwa-adv/visualwebarena")
if not Path(_VWA_ROOT, "browser_env").is_dir():
    import pytest
    pytest.skip(f"visualwebarena checkout not found at {_VWA_ROOT} (set VWA_ADV_REPO)",
                allow_module_level=True)
sys.path.insert(0, _VWA_ROOT)

# Real browser_env import needs these set before ANY import of
# environments.vwa_adv_tasks (which imports environments.vwa_adv_llm,
# which imports the real visualwebarena package) -- placeholders, no
# live server required for import-time or load_tasks()-level testing.
for _k, _v in [
    ("REDDIT", "http://PLACEHOLDER:8080"), ("SHOPPING", "http://PLACEHOLDER:8080"),
    ("WIKIPEDIA", "http://PLACEHOLDER:8080"), ("HOMEPAGE", "http://PLACEHOLDER:8080"),
    ("CLASSIFIEDS", "http://PLACEHOLDER:8080"), ("CLASSIFIEDS_RESET_TOKEN", "dummy"),
    ("REDDIT_RESET_URL", "http://PLACEHOLDER:8080"), ("SHOPPING_ADMIN", "http://PLACEHOLDER:8080"),
    ("GITLAB", "http://PLACEHOLDER:8080"), ("MAP", "http://PLACEHOLDER:8080"),
    ("OPENAI_API_KEY", "dummy"),
]:
    os.environ.setdefault(_k, _v)

from environments.vwa_adv_tasks import load_tasks, _early_stop, DEFAULT_EARLY_STOP_THRESHOLDS
from browser_env import ActionTypes, create_stop_action


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config_files_test"


def test_load_tasks_against_real_generated_configs():
    if not (CONFIG_DIR / "test_reddit").exists():
        print("  SKIP  (config_files_test/test_reddit not present in this environment)")
        return
    tasks = load_tasks(CONFIG_DIR, suite="test_reddit")
    assert len(tasks) == 210
    assert all(t.intent for t in tasks)
    assert all(t.config_file.exists() for t in tasks)
    # Ordered numerically, not lexicographically (so task_id "2" sorts
    # before "10", not after) -- confirms the sort key is doing its job.
    assert [t.task_id for t in tasks[:3]] == ["0", "1", "2"]


def test_load_tasks_missing_dir_raises_clear_error():
    raised = False
    try:
        load_tasks(Path("/nonexistent/path/xyz"), suite="test_reddit")
    except FileNotFoundError as e:
        raised = True
        assert "generate_test_data.py" in str(e)
    assert raised


def _click_action(element_id: str = "1"):
    return {"action_type": ActionTypes.CLICK, "element_id": element_id, "raw_prediction": f"click [{element_id}]"}


def _type_action(text: str, element_id: str = "1"):
    return {"action_type": ActionTypes.TYPE, "element_id": element_id, "text": list(text), "raw_prediction": f"type [{element_id}] [{text}]"}


def _none_action():
    a = create_stop_action("PARSE_FAILURE_PLACEHOLDER")
    a["action_type"] = ActionTypes.NONE
    return a


def _make_trajectory(actions: list) -> list:
    """Builds a Trajectory (alternating state_info/action, ENDING ON A
    STATE) from a list of actions -- represents N already-completed
    (action, state) pairs, i.e. exactly what early_stop() sees when
    called at the top of the loop deciding action N+1. Does NOT drop the
    trailing state: (len(trajectory)-1)/2 is the real num_steps formula,
    and it expects a matched state after the last action, not a dangling
    uncommitted one."""
    traj = [{"observation": {}, "info": {}}]
    for a in actions:
        traj.append(a)
        traj.append({"observation": {}, "info": {}})
    return traj


def test_early_stop_max_steps():
    traj = _make_trajectory([_click_action() for _ in range(5)])
    stop, reason = _early_stop(traj, max_steps=5, thresholds=DEFAULT_EARLY_STOP_THRESHOLDS)
    assert stop
    assert "max steps" in reason.lower()


def test_early_stop_not_triggered_under_max_steps():
    traj = _make_trajectory([_click_action() for _ in range(2)])
    stop, reason = _early_stop(traj, max_steps=30, thresholds=DEFAULT_EARLY_STOP_THRESHOLDS)
    assert not stop


def test_early_stop_parsing_failures():
    traj = _make_trajectory([_none_action() for _ in range(3)])
    stop, reason = _early_stop(traj, max_steps=30, thresholds=DEFAULT_EARLY_STOP_THRESHOLDS)
    assert stop
    assert "parse" in reason.lower()


def test_early_stop_repeating_non_type_action():
    traj = _make_trajectory([_click_action("42") for _ in range(3)])
    stop, reason = _early_stop(traj, max_steps=30, thresholds=DEFAULT_EARLY_STOP_THRESHOLDS)
    assert stop
    assert "same action" in reason.lower()


def test_early_stop_different_click_targets_not_flagged():
    traj = _make_trajectory([_click_action("1"), _click_action("2"), _click_action("3")])
    stop, reason = _early_stop(traj, max_steps=30, thresholds=DEFAULT_EARLY_STOP_THRESHOLDS)
    assert not stop


def test_early_stop_repeating_type_action_checked_across_whole_trajectory():
    """TYPE actions use a DIFFERENT branch (sum over the WHOLE action
    sequence, not just the last k) -- this is the exact distinction that
    made it worth fetching the real early_stop() in full rather than
    reimplementing from a partial read."""
    traj = _make_trajectory([
        _click_action("1"), _type_action("hello"), _click_action("2"), _type_action("hello"), _type_action("hello"),
    ])
    stop, reason = _early_stop(traj, max_steps=30, thresholds=DEFAULT_EARLY_STOP_THRESHOLDS)
    assert stop
    assert "typing" in reason.lower()


if __name__ == "__main__":
    tests = [
        test_load_tasks_against_real_generated_configs,
        test_load_tasks_missing_dir_raises_clear_error,
        test_early_stop_max_steps,
        test_early_stop_not_triggered_under_max_steps,
        test_early_stop_parsing_failures,
        test_early_stop_repeating_non_type_action,
        test_early_stop_different_click_targets_not_flagged,
        test_early_stop_repeating_type_action_checked_across_whole_trajectory,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
