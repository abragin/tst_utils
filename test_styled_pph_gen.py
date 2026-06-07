"""Unit tests for the 2A.4 Phase-1 filter logic and target-norm plumbing.

These are GPU-free: they exercise the pure-Python/numpy pieces of
``styled_pph_gen`` that decide *which rows survive* and *what norm the target
style embeddings get* — the two places a silent regression would corrupt the
downstream training data without any error.

Covered:
- ``phase1_keep_mask``: every threshold direction, the activated-only gender
  gate, boundary values, and the NaN / missing-column guards.
- ``_rescale_targets_to_source_norms`` / ``produce_target_style``: unit-norm
  (folder-14), legacy source-norm, and custom target_norm.
- ``save_tst_results``: the required-column assertion.
"""
import numpy as np
import pandas as pd
import pytest

from tst_utils.styled_pph_gen import (
    phase1_keep_mask,
    _rescale_targets_to_source_norms,
    produce_target_style,
    save_tst_results,
    CHRF_UB, BI_SCORE_UB, CL_SCORE_UB, GENDER_SCORE_LB,
    ENTITY_SCORE_LB, MEANING_SCORE_LB, SIM_MEASURE_UB, NAT_V2_LB,
)


def _passing_row(**over):
    """A row comfortably inside every keep threshold; override one to fail it."""
    base = dict(
        chrf=0.50, bi_score=0.05, cl_score=0.05,
        gender_score=0.90, gender_activated=True,
        entity_score=0.80, meaning_score=0.85,
        tst_result_style_sim=0.50, nat_v2=0.80,
    )
    base.update(over)
    return base


def _mask(*rows):
    return phase1_keep_mask(pd.DataFrame(list(rows))).to_numpy()


# ---------------- phase1_keep_mask: threshold directions ----------------

def test_clean_row_is_kept():
    assert _mask(_passing_row()).tolist() == [True]


@pytest.mark.parametrize("col,fail_val", [
    ("chrf", CHRF_UB),               # keep is strictly <, so == fails
    ("bi_score", BI_SCORE_UB),
    ("cl_score", CL_SCORE_UB),
    ("tst_result_style_sim", SIM_MEASURE_UB),
    ("entity_score", ENTITY_SCORE_LB - 0.01),
    ("meaning_score", MEANING_SCORE_LB),   # keep is strictly >, so == fails
    ("nat_v2", NAT_V2_LB - 0.01),
])
def test_each_filter_rejects(col, fail_val):
    assert _mask(_passing_row(**{col: fail_val})).tolist() == [False]


@pytest.mark.parametrize("col,keep_val", [
    ("entity_score", ENTITY_SCORE_LB),   # >= so boundary is kept
    ("nat_v2", NAT_V2_LB),               # >= so boundary is kept
])
def test_inclusive_lower_bounds_keep_boundary(col, keep_val):
    assert _mask(_passing_row(**{col: keep_val})).tolist() == [True]


# ---------------- gender gate is activated-only ----------------

def test_gender_rejects_only_when_activated():
    # low gender score but NOT activated -> gate does not fire -> kept
    not_activated = _passing_row(gender_score=0.1, gender_activated=False)
    activated = _passing_row(gender_score=0.1, gender_activated=True)
    assert _mask(not_activated).tolist() == [True]
    assert _mask(activated).tolist() == [False]


def test_gender_activated_above_threshold_kept():
    assert _mask(_passing_row(gender_score=GENDER_SCORE_LB,
                              gender_activated=True)).tolist() == [True]


# ---------------- NaN / missing-column guards ----------------

def test_nan_in_plain_column_raises():
    with pytest.raises(ValueError, match="NaN"):
        _mask(_passing_row(meaning_score=np.nan))


def test_nan_gender_score_on_nonactivated_is_ok():
    # NaN gender_score is legitimate when the pair is not gender-activated
    row = _passing_row(gender_score=np.nan, gender_activated=False)
    assert _mask(row).tolist() == [True]


def test_nan_gender_score_on_activated_raises():
    with pytest.raises(ValueError, match="gender_score"):
        _mask(_passing_row(gender_score=np.nan, gender_activated=True))


def test_missing_column_raises():
    df = pd.DataFrame([_passing_row()]).drop(columns=["nat_v2"])
    with pytest.raises(KeyError, match="nat_v2"):
        phase1_keep_mask(df)


def test_mixed_batch_keeps_only_clean_rows():
    mask = _mask(
        _passing_row(),                       # keep
        _passing_row(bi_score=0.5),           # reject (boilerplate)
        _passing_row(meaning_score=0.1),      # reject (meaning collapse)
        _passing_row(gender_score=0.1, gender_activated=False),  # keep
    )
    assert mask.tolist() == [True, False, False, True]


# ---------------- target_norm plumbing ----------------

def test_rescale_unit_norm():
    src = [np.array([3.0, 4.0]), np.array([0.0, 5.0])]       # norms 5, 5
    tgt = np.array([[1.0, 1.0], [2.0, 0.0]])
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=1.0)
    for v in out:
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_rescale_legacy_matches_source_norm():
    src = [np.array([3.0, 4.0]), np.array([6.0, 8.0])]       # norms 5, 10
    tgt = np.array([[1.0, 1.0], [1.0, 0.0]])
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=None)
    assert np.isclose(np.linalg.norm(out[0]), 5.0)
    assert np.isclose(np.linalg.norm(out[1]), 10.0)


def test_rescale_custom_norm():
    src = [np.array([1.0, 0.0])]
    tgt = np.array([[3.0, 4.0]])
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=15.0)
    assert np.isclose(np.linalg.norm(out[0]), 15.0)


def test_rescale_preserves_direction():
    src = [np.array([10.0, 0.0])]
    tgt = np.array([[3.0, 4.0]])                              # direction (0.6, 0.8)
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=1.0)
    assert np.allclose(out[0], [0.6, 0.8])


def _style_pool():
    rng = np.random.default_rng(0)
    rows = []
    for author in ["a", "b", "c"]:
        for _ in range(3):
            rows.append({"author": author,
                         "text_style_emb": rng.normal(size=8) * 7.0})
    return pd.DataFrame(rows)


def test_produce_target_style_unit_norm_for_folder14():
    pool = _style_pool()
    short = pool[pool.author == "a"].copy()
    out = produce_target_style(short, pool, key_col="author",
                               rng=np.random.default_rng(1), target_norm=1.0)
    for v in out:
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_produce_target_style_legacy_keeps_source_norm():
    pool = _style_pool()
    short = pool[pool.author == "a"].copy()
    src_norms = [np.linalg.norm(e) for e in short.text_style_emb]
    out = produce_target_style(short, pool, key_col="author",
                               rng=np.random.default_rng(1), target_norm=None)
    for v, n in zip(out, src_norms):
        assert np.isclose(np.linalg.norm(v), n)


# ---------------- save_tst_results required-column guard ----------------

def test_save_tst_results_missing_required_raises(tmp_path):
    df = pd.DataFrame([{"styled_text": "x"}])   # missing most required cols
    with pytest.raises(KeyError, match="required columns missing"):
        save_tst_results(df, str(tmp_path) + "/")
