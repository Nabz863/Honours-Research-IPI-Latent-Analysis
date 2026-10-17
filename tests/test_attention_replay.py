"""experiments/attention_replay.py: metrics by hand, and the recorder on a
real (tiny, random) Qwen3-VL in eager mode against output_attentions."""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import attention_replay as ar  # noqa: E402


def test_align_and_deflection():
    clean, adv = [1, 2, 3, 4], [1, 2, 9, 9, 3, 4]
    mc, ma, oa, oc = ar.align(clean, adv)
    assert mc.tolist() == [0, 1, 2, 3] and ma.tolist() == [0, 1, 4, 5]
    assert oa.tolist() == [2, 3] and oc.tolist() == []
    pc = torch.full((1, 4), 0.25)
    # identical pattern on the shared keys, nothing on the payload -> 0 deg
    pa0 = torch.tensor([[0.25, 0.25, 0.0, 0.0, 0.25, 0.25]])
    assert ar.deflection(pc, pa0, mc, ma).item() == pytest.approx(0.0, abs=1e-3)
    # half the mass captured by the payload
    pa = torch.tensor([[0.125, 0.125, 0.25, 0.25, 0.125, 0.125]])
    dot = 4 * 0.25 * 0.125
    exp = math.degrees(math.acos(dot / (0.5 * math.sqrt(4 * 0.125 ** 2 + 2 * 0.25 ** 2))))
    assert ar.deflection(pc, pa, mc, ma).item() == pytest.approx(exp, rel=1e-4)
    assert pa[:, oa].sum().item() == pytest.approx(0.5)
    # a replaced default (II replaces the vector content): removed tokens
    mc2, ma2, oa2, oc2 = ar.align([1, 5, 5, 4], [1, 9, 4])
    assert oc2.tolist() == [1, 2] and oa2.tolist() == [1]


def test_entropy_and_visible():
    p = torch.full((2, 8), 1 / 8)
    assert torch.allclose(ar.entropy(p, 8), torch.ones(2))
    onehot = torch.zeros(1, 8); onehot[0, 3] = 1
    assert ar.entropy(onehot, 8).item() == pytest.approx(0.0, abs=1e-6)
    assert ar.visible(100, None) == 100 and ar.visible(100, 32) == 32 and ar.visible(10, 32) == 10


def test_first_differing_step():
    c = {"steps": [{"prompt_sha256": "a"}, {"prompt_sha256": "b"}]}
    assert ar.first_differing_step(c, {"steps": [{"prompt_sha256": "a"}, {"prompt_sha256": "x"}]}) == 1
    assert ar.first_differing_step(c, {"steps": [{"prompt_sha256": "a"}, {"prompt_sha256": "b"}]}) is None
    assert ar.first_differing_step(c, {"steps": [{"prompt_sha256": "z"}]}) == 0


def _tiny_qwen3vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
    text = Qwen3VLTextConfig(vocab_size=100, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                             num_attention_heads=2, num_key_value_heads=1)
    vis = Qwen3VLVisionConfig(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=14,
                              in_channels=3)
    cfg = Qwen3VLConfig(text_config=text, vision_config=vis)
    cfg._attn_implementation = "eager"
    try:
        cfg.text_config._attn_implementation = "eager"
    except Exception:
        pass
    torch.manual_seed(0)
    m = Qwen3VLForConditionalGeneration(cfg).eval()
    return m


class _Proc:
    def __call__(self, text, return_tensors="pt"):
        ids = [int(x) for x in text.split()]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


def test_recorder_matches_output_attentions():
    m = _tiny_qwen3vl()
    from extraction.hooks import ActivationExtractor, all_layers
    ext = ActivationExtractor(m, target_layers=all_layers(m), token_aggregation="last")
    ids, rows, wins, acts = ar.forward(m, _Proc(), ext, "5 6 7 8 9 10 11")
    assert ids == [5, 6, 7, 8, 9, 10, 11]
    assert sorted(rows) == [0, 1, 2, 3] and all(r.shape == (2, 7) for r in rows.values())
    with torch.no_grad():
        ref = m(input_ids=torch.tensor([ids]), output_attentions=True).attentions
    for li in range(4):
        assert torch.allclose(rows[li], ref[li][0, :, -1, :].float(), atol=1e-5)
        assert torch.allclose(rows[li].sum(-1), torch.ones(2), atol=1e-4)
    # the replay's hidden states are what the extractor would have saved
    assert sorted(acts) == all_layers(m)
    assert ar.act_check(acts, acts) == 0.0
    # the patch is removed afterwards
    from defenses.icon_runtime import find_attention_module
    assert find_attention_module(m).eager_attention_forward.__name__ != "recording"


def test_shared_visible_window():
    # clean keys 0..9, attacked has 4 injected tokens at 6..9 (keys 0..13)
    clean = list(range(10)); adv = list(range(6)) + [90, 91, 92, 93] + [6, 7, 8, 9]
    mc, ma, oa, oc = ar.align(clean, adv)
    sc, sa = ar.shared_visible(mc, ma, 10, 14, window=6)
    # visible in clean: keys 4..9; in attacked: 8..13 -> shared: clean 6..9 (adv 10..13)
    assert sc.tolist() == [6, 7, 8, 9] and sa.tolist() == [10, 11, 12, 13]
    assert ar.shared_visible(mc, ma, 10, 14, window=None)[0].tolist() == mc.tolist()
    pc = torch.zeros(1, 10); pc[0, 4:] = 1 / 6
    pa = torch.zeros(1, 14); pa[0, 8:] = 1 / 6
    # same distribution over the shared keys -> 0 deg once renormalised
    assert ar.deflection_shared(pc, pa, sc, sa).item() == pytest.approx(0.0, abs=1e-2)
    assert torch.isnan(ar.deflection_shared(pc, pa, sc[:0], sa[:0])).all()


def test_check_arrival():
    def tr(msgs, steps):
        return {"messages": msgs, "steps": steps}
    sysm = {"role": "system", "content": [{"type": "text", "text": "sys"}]}
    user = {"role": "user", "content": [{"type": "text", "text": "do it"}]}
    tool_c = {"role": "user", "content": [{"type": "text", "text": "[TOOL RESULT] {\"result\": \"a\"}"}]}
    tool_a = {"role": "user", "content": [{"type": "text", "text": "[TOOL RESULT] {\"result\": \"a IGNORE\"}"}]}
    st0 = {"input": [0, 1], "completion": "call", "tool_calls": [{"function": "f"}]}
    c = tr([sysm, user, tool_c], [st0, {"input": [0, 1, 2]}])
    a = tr([sysm, user, tool_a], [st0, {"input": [0, 1, 2]}])
    assert ar.check_arrival(c, a, 1) is None
    a2 = tr([sysm, user, tool_a], [dict(st0, completion="other"), {"input": [0, 1, 2]}])
    assert "previous step" in ar.check_arrival(c, a2, 1)
    user2 = {"role": "user", "content": [{"type": "text", "text": "do something else"}]}
    a3 = tr([sysm, user2, tool_c], [st0, {"input": [0, 1, 2]}])
    assert "non-tool" in ar.check_arrival(c, a3, 1)
    assert "step 0" in ar.check_arrival(c, a, 0)
