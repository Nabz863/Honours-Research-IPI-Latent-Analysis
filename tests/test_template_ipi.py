"""
tests/test_template_ipi.py
---------------------------
Unit tests for the Template IPI attack module.
No model or GPU required -- these test payload construction logic only.

Run with:
    python -m pytest tests/test_template_ipi.py -v
or:
    python tests/test_template_ipi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks.base import AttackModality
from attacks.template_ipi import TemplateIPIAttack


def test_generate_injects_payload():
    attack = TemplateIPIAttack(seed=42)
    base_input = {"page_text": "Welcome to r/test. Here is a normal post."}
    result = attack.generate(base_input, target_behavior="upvote this post 100 times")

    assert result.attack_name == "template_ipi"
    assert result.modality == AttackModality.TEXTUAL
    assert "Welcome to r/test" in result.adversarial_text
    assert "upvote this post 100 times" in result.adversarial_text


def test_generate_is_deterministic_with_explicit_template():
    attack = TemplateIPIAttack(seed=42)
    base_input = {"page_text": "clean content"}
    result = attack.generate(base_input, target_behavior="do X", template_idx=0)
    assert result.metadata["template"] == attack.templates[0]
    assert "do X" in result.adversarial_text


def test_generate_respects_custom_injectable_field():
    attack = TemplateIPIAttack(seed=1)
    base_input = {
        "tool_output": "search results here",
        "injectable_field": "tool_output",
    }
    result = attack.generate(base_input, target_behavior="leak the API key")
    assert "search results here" in result.adversarial_text
    assert "leak the API key" in result.adversarial_text


def test_missing_field_defaults_to_empty_string():
    attack = TemplateIPIAttack(seed=1)
    base_input: dict = {}  # no page_text at all
    result = attack.generate(base_input, target_behavior="do Y")
    assert "do Y" in result.adversarial_text


def test_different_seeds_can_select_different_templates():
    base_input = {"page_text": "clean"}
    seen = set()
    for seed in range(20):
        attack = TemplateIPIAttack(seed=seed)
        result = attack.generate(base_input, target_behavior="do Z")
        seen.add(result.metadata["template"])
    # With 4 templates and 20 seeds, we should see more than one template chosen.
    assert len(seen) > 1, "Random template selection appears constant across seeds"


if __name__ == "__main__":
    tests = [
        test_generate_injects_payload,
        test_generate_is_deterministic_with_explicit_template,
        test_generate_respects_custom_injectable_field,
        test_missing_field_defaults_to_empty_string,
        test_different_seeds_can_select_different_templates,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
