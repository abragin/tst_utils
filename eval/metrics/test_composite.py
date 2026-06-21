import numpy as np
import pandas as pd
import pytest

from tst_utils.eval.metrics.composite import (
    AUTHOR_CE, DOMAIN_CE, base_score_v2, nat_v2_score, compute_nat_v2,
)
from tst_utils.eval.performance import TstPerformanceMetrics


# ---------------------------------------------------------------------------
# base_score_v2
# ---------------------------------------------------------------------------

def test_base_score_identity():
    assert base_score_v2(1.0, 1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_base_score_quality_floor():
    # gender=0, entity=0 → floor=0.10 applied; score well below a clean pair
    score = base_score_v2(1.0, 1.0, 1.0, 0.0, 0.0)
    assert 0.0 < score < 0.6


def test_base_score_vectorised():
    scores = base_score_v2(
        np.array([1.0, 0.5]),
        np.array([1.0, 0.5]),
        np.array([1.0, 0.5]),
        np.array([1.0, 0.5]),
        np.array([1.0, 0.5]),
    )
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0]


# ---------------------------------------------------------------------------
# nat_v2_score
# ---------------------------------------------------------------------------

def test_nat_v2_known_author_no_penalty():
    # styled CE well within Tolstoy range → no penalty
    assert nat_v2_score(6.0, 5.0, 'Tolstoy') == pytest.approx(1.0)


def test_nat_v2_known_author_with_penalty():
    # Expected derived from the imported AUTHOR_CE (robust to CE recalibration).
    # source below the author CE so tgt_ce_ref = author CE; styled well above it.
    ce = AUTHOR_CE['Tolstoy']
    styled, source = ce + 5.0, ce - 0.5
    ref = max(ce, source)  # = ce
    expected = 1.0 / ((styled - ref - 2.0) + 1)  # excess = 3.0 → 0.25
    result = nat_v2_score(styled, source, 'Tolstoy')
    assert result == pytest.approx(expected, abs=1e-3)
    assert result < 1.0  # a penalty actually fired


def test_nat_v2_source_ce_floor():
    # source_ce > author_ce → source_ce used as floor
    # styled=8.0, source=10.0, Tolstoy CE=6.637
    # tgt_ce_ref = max(6.637, 10.0) = 10.0
    # style_gap = 8.0 - 10.0 = -2.0 → no penalty
    assert nat_v2_score(8.0, 10.0, 'Tolstoy') == pytest.approx(1.0)


def test_nat_v2_unknown_author_falls_back_to_source_ce():
    # Unknown author: tgt_ce_ref = source_ce; style_gap = delta_CE
    # styled=6.0, source=5.0 → style_gap=1.0 < margin=2.0 → no penalty
    assert nat_v2_score(6.0, 5.0, 'UnknownAuthor') == pytest.approx(1.0)


def test_nat_v2_domain_fallback():
    # 'writers' is only in DOMAIN_CE (not AUTHOR_CE) → exercises the domain fallback.
    # Expected derived from the imported DOMAIN_CE (robust to CE recalibration).
    ce = DOMAIN_CE['writers']
    styled, source = ce + 4.0, ce - 0.5
    ref = max(ce, source)  # = ce
    expected = 1.0 / ((styled - ref - 2.0) + 1)  # excess = 2.0 → 1/3
    result = nat_v2_score(styled, source, 'writers')
    assert result == pytest.approx(expected, abs=1e-3)
    assert result < 1.0  # a penalty actually fired via the domain reference


def test_compute_nat_v2_vectorised():
    # Inputs framed relative to the imported AUTHOR_CE (robust to CE recalibration):
    #   row0: styled just above ref, gap < margin → no penalty
    #   row1: styled well above ref → penalty
    #   row2: source above ref → source floor → no penalty
    ce = AUTHOR_CE['Tolstoy']
    sty = [ce + 1.0, ce + 5.0, ce + 1.0]
    src = [ce - 1.0, ce - 1.0, ce + 3.0]
    styles = ['Tolstoy', 'Tolstoy', 'Tolstoy']
    results = compute_nat_v2(sty, src, styles)
    assert results[0] == pytest.approx(1.0)                           # no penalty
    assert results[1] == pytest.approx(1.0 / ((5.0 - 2.0) + 1), abs=1e-3)  # penalty, excess=3
    assert results[2] == pytest.approx(1.0)                           # source floor


# ---------------------------------------------------------------------------
# TstPerformanceMetrics.compute_composite_v2 — error handling
# ---------------------------------------------------------------------------

def _make_metrics_with_results(df):
    """Create a TstPerformanceMetrics instance with tst_results set, bypassing __init__."""
    m = TstPerformanceMetrics.__new__(TstPerformanceMetrics)
    m.tst_results = df
    return m


def test_compute_composite_v2_missing_score_cols():
    df = pd.DataFrame({
        'bi_score': [0.0], 'cl_score': [0.0], 'bi_cl': [1.0],
        'gender_score': [1.0], 'entity_score': [1.0],
        # missing: style_score, bert_score, labse_score, text_perplexity, styled_text_perplexity
    })
    with pytest.raises(ValueError, match='compute_scores'):
        _make_metrics_with_results(df).compute_composite_v2()


def test_compute_composite_v2_missing_quality_cols():
    df = pd.DataFrame({
        'style_score': [0.8], 'bert_score': [0.9], 'labse_score': [0.85],
        'text_perplexity': [5.0], 'styled_text_perplexity': [5.5],
        # missing: bi_score, cl_score, bi_cl, gender_score, entity_score
    })
    with pytest.raises(ValueError, match='compute_quality_scores'):
        _make_metrics_with_results(df).compute_composite_v2()


def test_compute_composite_v2_happy_path():
    # All inputs perfect, Tolstoy target: styled_CE=5.0, source_CE=5.0
    # tgt_ce_ref = max(6.637, 5.0) = 6.637; style_gap = 5.0 - 6.637 < 0 → nat_v2=1.0
    df = pd.DataFrame({
        'style_score': [1.0], 'bert_score': [1.0], 'labse_score': [1.0],
        'text_perplexity': [5.0], 'styled_text_perplexity': [5.0],
        'bi_score': [0.0], 'cl_score': [0.0], 'bi_cl': [1.0],
        'gender_score': [1.0], 'entity_score': [1.0],
        'target_style': ['Tolstoy'],
    })
    _make_metrics_with_results(df).compute_composite_v2()
    assert df['score_v2'].iloc[0] == pytest.approx(1.0)
    assert df['meaning_nolen'].iloc[0] == pytest.approx(1.0)
    assert df['nat_v2'].iloc[0] == pytest.approx(1.0)
