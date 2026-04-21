"""
Tests for calculate_perplexity and naturality_score.

Run on remote:
    ssh tallin.vpn 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tst311 && \
        cd /home/abragin/src/textprism/ && \
        pytest tst_utils/eval/metrics/test_naturality.py -v'
"""

import numpy as np
import pytest

from tst_utils.eval.metrics.naturality import naturality_score

_TEXTS = [
    'Привет мир это тест предложение.',
    'Я люблю читать книги каждый день.',
    'Он шёл по улице.',
]


@pytest.fixture(scope='module')
def perplexity_output():
    from tst_utils.eval.metrics.naturality import calculate_perplexity
    lls, ws = calculate_perplexity(_TEXTS, batch_size=4)
    return lls, ws


# ---------------------------------------------------------------------------
# calculate_perplexity
# ---------------------------------------------------------------------------

class TestCalculatePerplexity:
    def test_output_shapes(self, perplexity_output):
        lls, ws = perplexity_output
        assert lls.shape == (len(_TEXTS),)
        assert ws.shape == (len(_TEXTS),)

    def test_losses_positive(self, perplexity_output):
        lls, _ = perplexity_output
        assert (lls > 0).all()

    def test_weights_positive(self, perplexity_output):
        _, ws = perplexity_output
        assert (ws > 0).all()

    def test_aggregate_is_weighted_mean(self, perplexity_output):
        lls, ws = perplexity_output
        expected = float((lls * ws).sum() / ws.sum())
        from tst_utils.eval.metrics.naturality import calculate_perplexity
        agg = float(calculate_perplexity(_TEXTS, aggregate=True, batch_size=4))
        assert agg == pytest.approx(expected, rel=1e-3)

    def test_aggregate_within_loss_range(self, perplexity_output):
        lls, ws = perplexity_output
        agg = float((lls * ws).sum() / ws.sum())
        assert lls.min() <= agg <= lls.max()


# ---------------------------------------------------------------------------
# naturality_score
# ---------------------------------------------------------------------------

class TestNaturalityScore:
    def test_range(self):
        src = np.array([3.0, 5.0, 9.0])
        tgt = np.array([2.5, 4.0, 8.5])
        scores = naturality_score(src, tgt)
        assert scores.shape == (3,)
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_low_target_perplexity_gives_high_score(self):
        # target CE_loss well below 4 → both absolute and relative penalties near 0 → score near 1
        score = naturality_score(np.array([3.0]), np.array([2.0]))
        assert score > 0.9

    def test_high_target_perplexity_gives_low_score(self):
        # target CE_loss >> 4 → absolute penalty dominates → score near 0
        score = naturality_score(np.array([9.0]), np.array([9.5]))
        assert score < 0.2

    def test_target_below_source_beats_target_above_source(self):
        # same absolute target CE; one source is lower (so target exceeds it) → worse relative score
        tgt = np.array([5.5])
        score_below = naturality_score(np.array([6.0]), tgt)  # target < source
        score_above = naturality_score(np.array([4.0]), tgt)  # target > source
        assert score_below > score_above

    def test_identical_perplexity_zero_relative_penalty(self):
        # target == source → perpl_scaled_rel = 0 → perplexity_score_rel = 1
        src = np.array([5.0])
        score = naturality_score(src, src)
        # absolute penalty: 1/(8*(5-4)+1) = 1/9 ≈ 0.111; rel = 1 → score = sqrt(0.111) ≈ 0.333
        assert score == pytest.approx(np.sqrt(1 / 9), abs=1e-4)
