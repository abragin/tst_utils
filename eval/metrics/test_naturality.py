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

    def test_batch_size_invariance(self):
        """Per-text CE must not depend on batch composition.

        Regression test for the left-padding bug: rugpt3small's tokenizer pads on
        the left, and without position_ids GPT2 shifts real tokens' positional
        embeddings by the per-batch pad count, making CE depend on batch_size
        (observed up to ~7.6 CE units). Texts of very different lengths force
        substantial padding when batched together. bs=1 has no padding and is the
        correct reference; right-padding makes all batch sizes agree.
        See docs/issues/resolved/calculate-perplexity-left-pad-batch-dependence.md.
        """
        from tst_utils.eval.metrics.naturality import calculate_perplexity
        texts = [
            'Он шёл.',
            'Я люблю читать книги каждый день, особенно по вечерам, когда за окном идёт дождь.',
            'Привет.',
            'Москва — столица России, и в ней живёт очень много самых разных людей со всего света.',
        ]
        ref, _ = calculate_perplexity(texts, batch_size=1)      # no padding → correct
        batched, _ = calculate_perplexity(texts, batch_size=4)  # forces left/right padding
        assert np.abs(np.asarray(ref) - np.asarray(batched)).max() < 1e-3


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
