"""
Unit tests for GenderConsistencyScorer.

Run with:
    pytest tst_utils/eval/metrics/test_gender_consistency.py
"""
import pytest


@pytest.fixture(scope="module")
def scorer():
    from tst_utils.eval.metrics.gender_consistency import GenderConsistencyScorer
    return GenderConsistencyScorer(device="cuda")


# ── known-answer pairs ────────────────────────────────────────────────────────

SWITCH_THRESHOLD = 0.80


def test_clear_fem_to_masc_switch(scorer):
    r = scorer.score_pair(
        "Она написала письмо другу и отправила его.",
        "Он написал письмо другу и отправил его.",
    )
    assert r["has_switch"] is True
    assert r["activated"] is True
    assert r["gender_score"] < SWITCH_THRESHOLD


def test_clear_masc_to_fem_switch(scorer):
    r = scorer.score_pair(
        "Он пошёл домой и лёг спать.",
        "Она пошла домой и легла спать.",
    )
    assert r["has_switch"] is True
    assert r["gender_score"] < SWITCH_THRESHOLD


def test_first_person_switch(scorer):
    r = scorer.score_pair(
        "Я написал отчёт и отправил его.",
        "Я написала отчёт и отправила его.",
    )
    assert r["has_switch"] is True
    assert r["activated"] is True


def test_no_switch_identical(scorer):
    src = "Она написала письмо другу."
    r = scorer.score_pair(src, src)
    assert r["has_switch"] is False
    assert r["gender_score"] == pytest.approx(1.0)


def test_no_switch_same_gender_rephrased(scorer):
    r = scorer.score_pair(
        "Он пошёл домой.",
        "Он вернулся домой.",
    )
    assert r["has_switch"] is False
    assert r["gender_score"] == pytest.approx(1.0)


def test_no_gendered_tokens_returns_no_switch(scorer):
    """Texts with no PRON+Sing → not activated, score=1.0."""
    r = scorer.score_pair(
        "Птицы пели над рекой весенним утром.",
        "Птицы щебетали над рекой тёплым утром.",
    )
    assert r["activated"] is False
    assert r["has_switch"] is False
    assert r["score_A"] == pytest.approx(1.0)


def test_inanimate_subject_no_switch(scorer):
    """Inanimate subject (no pronoun) should not activate the metric."""
    r = scorer.score_pair(
        "В зале стояла тишина.",
        "В зале воцарилась тишина.",
    )
    assert r["activated"] is False
    assert r["has_switch"] is False


def test_empty_source_returns_no_switch(scorer):
    r = scorer.score_pair("", "Она написала письмо.")
    assert r["has_switch"] is False
    assert r["gender_score"] == pytest.approx(1.0)


def test_empty_target_returns_no_switch(scorer):
    r = scorer.score_pair("Она написала письмо.", "")
    assert r["has_switch"] is False
    assert r["gender_score"] == pytest.approx(1.0)


# ── return type checks ────────────────────────────────────────────────────────

def test_score_pair_keys(scorer):
    r = scorer.score_pair("Она написала.", "Он написал.")
    assert set(r.keys()) == {"gender_score", "score_A", "score_B", "activated", "has_switch"}


def test_score_pair_types(scorer):
    r = scorer.score_pair("Она написала.", "Он написал.")
    assert isinstance(r["gender_score"], float)
    assert isinstance(r["score_A"], float)
    assert isinstance(r["score_B"], float)
    assert isinstance(r["activated"], bool)
    assert isinstance(r["has_switch"], bool)


def test_score_pair_range(scorer):
    for src, tgt in [
        ("Она написала письмо.", "Он написал письмо."),
        ("Он пошёл домой.", "Он вернулся домой."),
        ("Птицы пели.", "Птицы щебетали."),
    ]:
        r = scorer.score_pair(src, tgt)
        assert 0.0 <= r["gender_score"] <= 1.0
        assert 0.0 <= r["score_A"] <= 1.0
        assert 0.0 <= r["score_B"] <= 1.0


def test_score_batch_shape(scorer):
    import numpy as np
    sources = ["Она написала.", "Он пошёл.", "Птицы пели."]
    targets = ["Он написал.", "Он вернулся.", "Птицы щебетали."]
    batch = scorer.score_batch(sources, targets)
    assert set(batch.keys()) == {"gender_score", "score_A", "score_B", "activated", "has_switch"}
    for key in batch:
        assert len(batch[key]) == 3
    assert batch["gender_score"].dtype in (np.float64, np.float32)
    assert batch["has_switch"].dtype == bool


def test_score_batch_matches_score_pair(scorer):
    """score_batch must agree with individual score_pair calls."""
    pairs = [
        ("Она написала письмо.", "Он написал письмо."),
        ("Он пошёл домой.", "Он вернулся домой."),
    ]
    batch = scorer.score_batch([s for s, _ in pairs], [t for _, t in pairs])
    for i, (src, tgt) in enumerate(pairs):
        r = scorer.score_pair(src, tgt)
        assert batch["gender_score"][i] == pytest.approx(r["gender_score"])
