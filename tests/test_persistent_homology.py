"""analysis/persistent_homology.py: the self-contained H0 (union-find)
persistence, the Betti curve, the summaries, the clean-vs-attack Betti-L2
permutation test (separable vs null), and the cloud loader. ripser is not
required for these; H1 assertions are guarded behind HAVE_RIPSER."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import persistent_homology as ph  # noqa: E402

L = 31


def test_h0_union_find_known():
    # three points on a line: 0, 0.1, 5.0 -> merges at 0.1 then 4.9, + inf
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])
    dg = ph._h0_union_find(pts)
    finite = sorted(float(d) for _, d in dg if np.isfinite(d))
    assert finite == pytest.approx([0.1, 4.9], abs=1e-6)
    assert np.isinf(dg[:, 1]).sum() == 1          # exactly one infinite component


def test_summarise_drops_infinite():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])
    s = ph.summarise(ph._h0_union_find(pts))
    assert s["n"] == 2
    assert s["total_persistence"] == pytest.approx(5.0, abs=1e-6)
    assert s["max_persistence"] == pytest.approx(4.9, abs=1e-6)


def test_betti_curve_monotone():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])
    dg = ph._h0_union_find(pts)
    radii = np.array([0.0, 0.05, 0.2, 5.0])
    bc = ph.betti_curve(dg, radii)
    assert list(bc) == [3.0, 3.0, 2.0, 1.0]       # components merge as r grows


def test_diagrams_fallback_shape():
    pts = np.random.default_rng(0).normal(size=(8, 3))
    dg = ph.diagrams(pts, maxdim=1)
    assert 0 in dg and dg[0].shape[1] == 2
    if not ph.HAVE_RIPSER:
        assert dg[1] is None


def test_compare_separable_vs_null():
    rng = np.random.default_rng(1)
    # tight clean cloud vs a much more DISPERSED cloud -> different H0 shape
    clean = rng.normal(scale=0.2, size=(24, 2))
    disp = rng.normal(scale=4.0, size=(24, 2))
    radii = ph._radius_grid(clean, disp, 50)
    res = ph.compare(clean, disp, radii, n_perm=200, maxdim=1, rng=rng)
    h0 = res["betti_l2"][0]
    assert h0["observed"] > h0["null_mean"]        # true split is more extreme
    assert h0["p_perm"] is not None and h0["p_perm"] <= 0.05

    # same distribution twice -> not significant
    a = rng.normal(scale=1.0, size=(24, 2))
    b = rng.normal(scale=1.0, size=(24, 2))
    radii2 = ph._radius_grid(a, b, 50)
    res2 = ph.compare(a, b, radii2, n_perm=200, maxdim=1, rng=rng)
    assert res2["betti_l2"][0]["p_perm"] > 0.05


def _ep(d, sid, vecs):
    """vecs: list of per-turn layer-L vectors (mean-over-turns in the loader)."""
    torch.save([{L: torch.tensor(v, dtype=torch.float32)} for v in vecs],
               d / f"{sid}.pt")


def test_clouds_loader(tmp_path):
    d = tmp_path / "final_x"
    d.mkdir()
    rows = []
    rng = np.random.default_rng(3)
    for i in range(6):
        tid = f"workspace_user_task_{i}_injection_task_0"
        cv = rng.normal(size=4)
        _ep(d, f"clean_{tid}", [cv, cv + 0.1])         # 2 turns -> mean
        rows.append({"sample_id": f"clean_{tid}", "condition": "clean",
                     "attack_name": "clean", "task_id": tid, "suite_name": "workspace",
                     "defense_name": None, "error": None})
        _ep(d, f"tipi_{tid}", [cv + 3.0])
        rows.append({"sample_id": f"tipi_{tid}", "condition": "attacked",
                     "attack_name": "template_ipi", "task_id": tid, "suite_name": "workspace",
                     "defense_name": None, "error": None})
    # a defended row must be ignored, and an errored attacked row dropped
    rows.append({"sample_id": "def_x", "condition": "attacked", "attack_name": "template_ipi",
                 "task_id": "t", "suite_name": "workspace", "defense_name": "icon", "error": None})
    (d / "attack_results.json").write_text(json.dumps(rows))

    cl = ph.clouds(str(d), L, "mean")
    assert set(cl) == {"clean", "template_ipi"}
    assert cl["clean"].shape == (6, 4) and cl["template_ipi"].shape == (6, 4)


def test_preprocess_dim_bound():
    rng = np.random.default_rng(4)
    clean = rng.normal(size=(5, 20))
    other = rng.normal(size=(5, 20))
    _, oz, info = ph.preprocess(clean, other, pca_dim=50, seed=0)
    # k <= n_pooled - 1 = 9
    assert info["pca_dim"] <= 9 and oz.shape[1] == info["pca_dim"]


# ---------------------------------------------------------------------------
# Regression D (finding 6): without ripser H1 is UNAVAILABLE and must be
# absent from the output -- never a fabricated observed=0.0, p=1.0.
# ---------------------------------------------------------------------------

def test_D_no_ripser_drops_H1_instead_of_fabricating_it(monkeypatch):
    monkeypatch.setattr(ph, "HAVE_RIPSER", False)
    monkeypatch.setattr(ph, "HAVE_PERSIM", False)
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(12, 3)), rng.normal(size=(12, 3)) + 2.0
    res = ph.compare(a, b, ph._radius_grid(a, b, 20), n_perm=20, maxdim=1, rng=rng)
    assert 0 in res["betti_l2"]
    assert 1 not in res["betti_l2"], "H1 fabricated without ripser"
    assert 1 not in res["summaries"]["clean"] and 1 not in res["summaries"]["other"]


@pytest.mark.skipif(not ph.HAVE_RIPSER, reason="needs ripser")
def test_D_with_ripser_H1_is_reported():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=(12, 3)), rng.normal(size=(12, 3)) + 2.0
    res = ph.compare(a, b, ph._radius_grid(a, b, 20), n_perm=10, maxdim=1, rng=rng)
    assert 1 in res["betti_l2"]


# ---------------------------------------------------------------------------
# Regression E (finding 7): the attacked cloud must hold only episodes where
# the injection entered the representation (||a - c|| > ZERO_TOL against the
# SAME task's clean episode), and identical points are collapsed. On AgentDojo
# ~83% of attacked episodes are byte-identical to clean, and the clean episode
# recurs once per injection task, so without this both clouds are mostly
# duplicates and H0 is dominated by zero-length deaths.
# ---------------------------------------------------------------------------

def _row_e(rows, sid, cond, attack, tid, error=None):
    rows.append({"sample_id": sid, "condition": cond, "attack_name": attack,
                 "task_id": tid, "suite_name": "workspace", "defense_name": None,
                 "error": error})


def _e_dir(tmp_path):
    d = tmp_path / "final_e"
    d.mkdir()
    rows = []
    rng = np.random.default_rng(7)
    cv = {u: rng.normal(size=4) for u in range(4)}       # clean depends on user task only
    for u in range(3):
        for j in range(4):
            tid = f"workspace_user_task_{u}_injection_task_{j}"
            _ep(d, f"clean_{tid}", [cv[u]])               # bit-identical across j
            _row_e(rows, f"clean_{tid}", "clean", "clean", tid)
            if j < 2:
                av = cv[u]                                # never reached: == clean
            elif u == 0:
                av = cv[u] + 5.0                          # u0: j=2,3 identical reached points
            else:
                av = cv[u] + rng.normal(size=4)
            _ep(d, f"tipi_{tid}", [av])
            _row_e(rows, f"tipi_{tid}", "attacked", "template_ipi", tid)
    # user task 3: clean errored -> its attacked episode has no partner and
    # cannot be shown to be reached, so it is excluded
    tid = "workspace_user_task_3_injection_task_0"
    _ep(d, f"clean_{tid}", [cv[3]])
    _row_e(rows, f"clean_{tid}", "clean", "clean", tid, error="timeout")
    _ep(d, f"tipi_{tid}", [cv[3] + 9.0])
    _row_e(rows, f"tipi_{tid}", "attacked", "template_ipi", tid)
    # a near-duplicate of a clean point (float noise below ZERO_TOL) is the same point
    tid = "workspace_user_task_1_injection_task_9"
    _ep(d, f"clean_{tid}", [cv[1] + 1e-9])
    _row_e(rows, f"clean_{tid}", "clean", "clean", tid)
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d, cv


def test_E_clouds_drop_zero_delta_and_duplicates(tmp_path):
    d, cv = _e_dir(tmp_path)
    cl = ph.clouds(str(d), L, "mean")
    # clean: 13 loaded rows, 3 distinct points
    assert cl["clean"].shape == (3, 4)
    # attacked: 13 rows - 6 zero-delta - 1 without a clean partner = 6 reached,
    # of which u0's two are identical -> 5 points
    assert cl["template_ipi"].shape == (5, 4)
    # no attacked point coincides with any clean point
    dist = np.linalg.norm(cl["template_ipi"][:, None, :] - cl["clean"][None, :, :], axis=-1)
    assert dist.min() > 1e-3


def test_E_clouds_report_what_was_excluded(tmp_path):
    d, _ = _e_dir(tmp_path)
    cl, info = ph.clouds(str(d), L, "mean", return_info=True)
    assert info["clean"] == {"n_loaded": 13, "n_unique": 3}
    assert info["template_ipi"] == {"n_loaded": 13, "n_no_clean": 1,
                                    "n_zero_delta": 6, "n_reached": 6, "n_unique": 5}


# --- review finding 8b: a directory with no analysable attack must not PASS --

def test_main_exits_nonzero_when_no_attack_is_analysable(tmp_path, monkeypatch):
    d = tmp_path / "final_none"
    d.mkdir()
    rows = []
    rng = np.random.default_rng(9)
    for i in range(12):
        tid = f"workspace_user_task_{i}_injection_task_0"
        cv = rng.normal(size=4)
        _ep(d, f"clean_{tid}", [cv])
        _row_e(rows, f"clean_{tid}", "clean", "clean", tid)
        _ep(d, f"tipi_{tid}", [cv])                       # never reached
        _row_e(rows, f"tipi_{tid}", "attacked", "template_ipi", tid)
    (d / "attack_results.json").write_text(json.dumps(rows))
    out = tmp_path / "o" / "h.json"
    monkeypatch.setattr(sys, "argv", ["persistent_homology.py", "--results_dir", str(d),
                                      "--layer", str(L), "--n_perm", "5", "--out", str(out)])
    with pytest.raises(SystemExit) as ex:
        ph.main()
    assert ex.value.code not in (0, None)
    res = json.loads(out.read_text())                     # written before the exit
    assert res["per_attack"] == {} and res["clouds"]["template_ipi"]["n_zero_delta"] == 12


# ---------------------------------------------------------------------------
# Review finding 1 (RQ3 null): the test compares one clean and one attacked
# point per REACHED user task under a within-task swap null. Comparing the full
# clean cloud with the reached attacked cloud under a free label split
# "detects" which tasks were reached (p=0.01 with no effect at all).
# ---------------------------------------------------------------------------

def _rq3_dir(d, seed, effect=None, n_users=30, n_reached=15, n_inj=3, dim=16):
    d.mkdir()
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        c = rng.normal(size=dim)                      # clean depends on the user task only
        for j in range(n_inj):
            tid = f"workspace_user_task_{u}_injection_task_{j}"
            _ep(d, f"clean_{tid}", [c])
            _row_e(rows, f"clean_{tid}", "clean", "clean", tid)
            if u >= n_reached:
                a = c                                 # never reached
            elif effect is None:
                a = c + 0.02 * rng.normal(size=dim)   # reached, no real change
            else:
                a = effect(c, rng)
            _ep(d, f"tipi_{tid}", [a])
            _row_e(rows, f"tipi_{tid}", "attacked", "template_ipi", tid)
    (d / "attack_results.json").write_text(json.dumps(rows))
    return d


def _run_main(d, out, monkeypatch, n_perm=100):
    monkeypatch.setattr(sys, "argv", ["persistent_homology.py", "--results_dir", str(d),
                                      "--layer", str(L), "--n_perm", str(n_perm),
                                      "--min_points", "10", "--out", str(out)])
    ph.main()
    return json.loads(out.read_text())["per_attack"]["template_ipi"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rq3_no_change_is_not_significant(tmp_path, monkeypatch, seed):
    res = _run_main(_rq3_dir(tmp_path / "d", seed), tmp_path / "o.json", monkeypatch)
    assert res["preprocess"]["n_pairs"] == 15            # one pair per reached user task
    p = res["betti_l2"]["0"]["p_perm"]
    assert p is None or p > 0.05        # None: curves identical under every swap


def test_rq3_detects_a_real_change(tmp_path, monkeypatch):
    res = _run_main(_rq3_dir(tmp_path / "d", 0, effect=lambda c, rng: 3.0 * c),
                    tmp_path / "o.json", monkeypatch)
    assert res["betti_l2"]["0"]["p_perm"] <= 0.05


def test_matched_pairs_one_per_reached_user_task(tmp_path):
    d = _rq3_dir(tmp_path / "d", 4, n_users=6, n_reached=4, n_inj=3, dim=3)
    cbt, _, att = ph._load_undefended(str(d), L, "mean")
    C, A, groups = ph.matched_pairs(cbt, att)["template_ipi"]
    assert groups == [f"workspace_user_task_{u}" for u in range(4)]
    for i, g in enumerate(groups):                       # row i is user task i, both sides
        tids = [t for t, _ in att["template_ipi"] if t.startswith(g + "_")]
        assert np.allclose(C[i], np.mean([cbt[t] for t in tids], axis=0))
        assert np.allclose(A[i], np.mean([v for t, v in att["template_ipi"] if t in tids], axis=0))


def test_paired_null_only_swaps_within_pairs():
    # every null draw must be a per-row swap of (C, A): the pooled multiset of
    # each row is preserved. Instrument diagrams() to record what it is given.
    rng = np.random.default_rng(0)
    C, A = rng.normal(size=(8, 2)), rng.normal(size=(8, 2)) + 5.0
    seen = []
    real = ph.diagrams
    def spy(points, maxdim=1):
        seen.append(np.array(points))
        return real(points, maxdim)
    ph.diagrams = spy
    try:
        ph.compare(C, A, ph._radius_grid(C, A, 10), n_perm=5, maxdim=0, rng=rng, paired=True)
    finally:
        ph.diagrams = real
    for g1, g2 in zip(seen[2::2], seen[3::2]):
        for i in range(8):
            assert {tuple(g1[i]), tuple(g2[i])} == {tuple(C[i]), tuple(A[i])}


def test_main_uses_the_paired_null_and_pooled_scaling(tmp_path, monkeypatch):
    """Under H0 a free split or clean-only scaling still look calibrated, so pin
    the design itself: main must call preprocess(scale_on='pooled') (label
    invariant -> the swap null is exact) and compare(paired=True)."""
    calls = {}
    real_pre, real_cmp = ph.preprocess, ph.compare
    def pre(*a, **k):
        calls["scale_on"] = k.get("scale_on", "clean")
        return real_pre(*a, **k)
    def cmp(*a, **k):
        calls["paired"] = k.get("paired", False)
        return real_cmp(*a, **k)
    real_grid = ph._radius_grid
    def grid(*a, **k):
        calls["grid_paired"] = k.get("paired", False)
        return real_grid(*a, **k)
    monkeypatch.setattr(ph, "preprocess", pre)
    monkeypatch.setattr(ph, "compare", cmp)
    monkeypatch.setattr(ph, "_radius_grid", grid)
    _run_main(_rq3_dir(tmp_path / "d", 0), tmp_path / "o.json", monkeypatch, n_perm=5)
    assert calls == {"scale_on": "pooled", "paired": True, "grid_paired": True}


# --- RQ3 radius grid must span the merges (review: flat curves -> p = 1) -----

def test_radius_grid_spans_all_h0_deaths_one_point_per_task_high_dim():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(20, 64))
    A = 1.5 * C + rng.normal(size=(20, 64))
    cz, oz, _ = ph.preprocess(C, A, 50, 0, scale_on="pooled")
    radii = ph._radius_grid(cz, oz, 50)
    for cloud in (cz, oz):
        deaths = ph._h0_union_find(cloud)[:, 1]
        deaths = deaths[np.isfinite(deaths)]
        assert radii[0] <= deaths.min() and deaths.max() <= radii[-1]
    res = ph.compare(cz, oz, radii, n_perm=100, maxdim=0, rng=rng, paired=True)
    assert res["betti_l2"][0]["observed"] > 0
    assert res["betti_l2"][0]["p_perm"] <= 0.05


def test_radius_grid_and_pooled_scaling_are_swap_invariant():
    rng = np.random.default_rng(1)
    C, A = rng.normal(size=(9, 5)), 2.0 * rng.normal(size=(9, 5)) + 1.0
    assert np.allclose(ph._radius_grid(C, A, 20), ph._radius_grid(A, C, 20))
    c1, a1, _ = ph.preprocess(C, A, 4, 0, scale_on="pooled")
    a2, c2, _ = ph.preprocess(A, C, 4, 0, scale_on="pooled")
    assert np.allclose(np.abs(c1), np.abs(c2)) and np.allclose(np.abs(a1), np.abs(a2))


def test_identical_clouds_give_no_pvalue_not_p1():
    rng = np.random.default_rng(2)
    C = rng.normal(size=(10, 3))
    res = ph.compare(C, C.copy(), ph._radius_grid(C, C, 20), n_perm=20, maxdim=0,
                     rng=rng, paired=True)
    assert res["betti_l2"][0]["p_perm"] is None


def test_main_subsamples_pairs_together(tmp_path, monkeypatch):
    seen = {}
    real_pre, real_cmp = ph.preprocess, ph.compare
    def pre(*a, **k):
        seen["pre"] = real_pre(*a, **k)
        return seen["pre"]
    def cmp(c, o, *a, **k):
        seen["cmp"] = (c, o)
        return real_cmp(c, o, *a, **k)
    monkeypatch.setattr(ph, "preprocess", pre)
    monkeypatch.setattr(ph, "compare", cmp)
    d = _rq3_dir(tmp_path / "d", 3, n_users=30, n_reached=24)
    monkeypatch.setattr(sys, "argv", ["persistent_homology.py", "--results_dir", str(d),
                                      "--layer", str(L), "--n_perm", "5", "--max_points", "12",
                                      "--out", str(tmp_path / "o.json")])
    ph.main()
    cz, oz, _ = seen["pre"]
    c2, o2 = seen["cmp"]
    assert len(c2) == 12
    for i in range(12):
        j = int(np.argmin(np.linalg.norm(cz - c2[i], axis=1)))
        assert np.allclose(cz[j], c2[i]) and np.allclose(oz[j], o2[i])   # same task row


def test_paired_grid_skips_within_pair_distances():
    rng = np.random.default_rng(5)
    C = rng.normal(size=(20, 8))
    A = C + 1e-3 * rng.normal(size=(20, 8))          # tiny attack displacement
    radii = ph._radius_grid(C, A, 50, paired=True)
    assert radii[0] > 0.1                              # not wasted near |a_i - c_i| ~ 1e-3
    for cloud in (C, A):                               # still spans every sub-cloud merge
        dth = ph._h0_union_find(cloud)[:, 1]
        dth = dth[np.isfinite(dth)]
        assert radii[0] <= dth.min() and dth.max() <= radii[-1]
