"""Tests for the batched GenderConsistencyScorer (task 2A.3.1).

score_batch now batches the SimAlign alignment (BatchedAligner.align_batch) and
caches source-side morphology across a source's N targets, instead of a per-pair
loop. It must produce output identical to the per-pair score_pair loop.

Runs on CPU (mBERT + rupostagger) when GPU is absent; skips if deps are missing.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("simalign")
pytest.importorskip("rupostagger")

from tst_utils.eval.metrics.gender_consistency import GenderConsistencyScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 1e-9

# A few pairs incl. a gender switch, a clean copy, a reused source across targets.
SRC = [
    "Она написала письмо и ушла домой.",
    "Она написала письмо и ушла домой.",
    "Мальчик играл во дворе целый день.",
    "Врач осмотрел пациента утром.",
]
TGT = [
    "Он написал письмо и ушёл домой.",      # switch
    "Она написала письмо и ушла домой.",    # identical
    "Ребёнок играл во дворе весь день.",
    "Доктор осмотрел больного утром.",
]


@pytest.fixture(scope="module")
def scorer():
    # rupostagger downloads its model on load() (via gdown); skip cleanly where the
    # model can't be fetched (e.g. offline box) — this is a model-availability issue,
    # not a GPU one.
    try:
        return GenderConsistencyScorer(device=DEVICE)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"GenderConsistencyScorer unavailable (model/dep load failed): {e}")


def test_batch_matches_perpair(scorer):
    """score_batch == per-pair score_pair loop, field by field."""
    batched = scorer.score_batch(SRC, TGT)
    perpair = [scorer.score_pair(s, t) for s, t in zip(SRC, TGT)]
    for key in ("gender_score", "score_A", "score_B"):
        a = batched[key]
        b = np.array([p[key] for p in perpair])
        assert np.abs(a - b).max() < TOL, f"{key} mismatch"
    assert list(batched["activated"]) == [p["activated"] for p in perpair]
    assert list(batched["has_switch"]) == [p["has_switch"] for p in perpair]


def test_source_reuse_invariance(scorer):
    """One source reused across many targets is scored consistently regardless of
    batching order (exercises the source-morphology cache + align_batch reuse)."""
    src = [SRC[0]] * 3 + [SRC[2]]
    tgt = [TGT[0], TGT[1], TGT[0], TGT[2]]
    out = scorer.score_batch(src, tgt)
    # the two identical (src,tgt)=(SRC0,TGT0) rows must get identical scores
    assert abs(out["gender_score"][0] - out["gender_score"][2]) < TOL


def test_empty_inputs(scorer):
    out = scorer.score_batch(["", "Она ушла."], ["Она ушла.", ""])
    assert out["gender_score"][0] == 1.0 and out["score_A"][0] == 1.0
    assert out["gender_score"][1] == 1.0
    assert np.all(np.isfinite(out["gender_score"]))
