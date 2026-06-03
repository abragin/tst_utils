"""Tests for BatchedAligner (task 2A.3.1).

BatchedAligner is a batched, source-cached, model-agnostic drop-in for
AlignmentScorer (BatchedMBertAligner is a backward-compat alias).

Two tiers so the suite is meaningful WITHOUT a GPU:

  - **CPU tests** (model-agnostic orchestration): batched==per-pair parity,
    source-reuse / order invariance, overflow→fallback routing, edge cases,
    return shape. These use the tiny `cointegrated/rubert-tiny2` (3-layer) on
    whatever device is available (CPU is fine and fast), comparing against an
    in-test per-pair SimAlign reference — self-contained, deterministic, no GPU.

  - **GPU tests** (exact large-model parity): the frozen mBERT-L9 golden and the
    ruRoberta production-eval drop-in. These need the specific large models and
    exact values, so they skip when CUDA is unavailable (run on tallin).

    pytest tst_utils/eval/metrics/tests/test_batched_aligner.py        # CPU subset runs anywhere
"""

import gc
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("simalign")

from tst_utils.eval.metrics.alignment import (
    AlignmentScorer, BatchedAligner, BatchedMBertAligner)

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
gpu_only = pytest.mark.skipif(
    not HAS_CUDA, reason="exact large-model parity — run on tallin GPU")

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_mbert_l9_bicl.json")
GOLDEN_RUROBERTA = os.path.join(os.path.dirname(__file__), "golden_ruroberta_l8_bicl.json")
TOL = 1e-6
TINY = "cointegrated/rubert-tiny2"   # 3-layer; CPU-fast; for model-agnostic logic
TINY_LAYER = 3


@pytest.fixture(scope="module")
def golden():
    return json.load(open(GOLDEN))


@pytest.fixture(scope="module")
def texts(golden):
    # plain text pairs (model-agnostic) — usable by both CPU and GPU tests
    return golden["texts"], golden["styled_texts"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _score(unaligned, total, score_fn, window, min_span):
    if total == 0 or not unaligned:
        return 0.0
    if score_fn == "rolling_max":
        return float(np.max(AlignmentScorer.rolling_density_map(unaligned, total, window)))
    if min_span <= 1:
        return len(unaligned) / total
    idx = sorted(unaligned); cnt, run = 0, 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or idx[k] != idx[k - 1] + 1:
            if k - run >= min_span:
                cnt += k - run
            run = k
    return cnt / total


def _perpair(sa, src, tgt, score_fn="rolling_max", window=9, min_span=4):
    """Per-pair SimAlign reference using SentenceAligner `sa`."""
    sw = [BatchedAligner._clean_word(w) for w in src.split()]
    tw = [BatchedAligner._clean_word(w) for w in tgt.split()]
    if not sw or not tw:
        return 0.0, 0.0
    al = sa.get_word_aligns(sw, tw)["itermax"]
    asrc = {a for a, b in al}; atgt = {b for a, b in al}
    bi = [j for j in range(len(tw)) if j not in atgt]
    cl = [i for i in range(len(sw)) if i not in asrc]
    return (_score(bi, len(tw), score_fn, window, min_span),
            _score(cl, len(sw), score_fn, window, min_span))


@pytest.fixture(scope="module")
def tiny_perpair_sa():
    from simalign import SentenceAligner
    return SentenceAligner(model=TINY, token_type="word", matching_methods="i",
                           device=DEVICE, layer=TINY_LAYER)


# ── CPU tests (model-agnostic orchestration; tiny model, any device) ────────────
def test_batched_matches_perpair(texts, tiny_perpair_sa):
    """BatchedAligner reproduces a per-pair SimAlign reference (rolling_max + span)."""
    src, tgt = texts
    src, tgt = src[:12], tgt[:12]
    for fn, kw in [("rolling_max", dict(window=9)), ("span", dict(min_span=4))]:
        al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn=fn, device=DEVICE,
                            enc_batch_size=8, **kw)
        out = al.score_batch(src, tgt, return_masks=False)
        for i, (s, t) in enumerate(zip(src, tgt)):
            rb, rc = _perpair(tiny_perpair_sa, s, t, score_fn=fn,
                              window=kw.get("window", 9), min_span=kw.get("min_span", 4))
            assert abs(out["bi_score"][i] - rb) < TOL
            assert abs(out["cl_score"][i] - rc) < TOL


def test_order_and_source_reuse_invariance(texts):
    """Shuffled rows + one source reused across many targets give identical scores."""
    src, tgt = texts
    shared = src[0]
    src2 = [shared] * 20 + list(src[:20])
    tgt2 = list(tgt[:20]) + list(tgt[:20])
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="rolling_max",
                        window=9, device=DEVICE)
    canon = al.score_batch(src2, tgt2, return_masks=False)
    perm = np.random.RandomState(0).permutation(len(src2))
    sh = al.score_batch([src2[i] for i in perm], [tgt2[i] for i in perm], return_masks=False)
    assert np.abs(sh["bi_score"] - canon["bi_score"][perm]).max() < TOL
    assert np.abs(sh["cl_score"] - canon["cl_score"][perm]).max() < TOL


def test_overflow_routes_to_fallback(texts, tiny_perpair_sa):
    """Pairs over the (here forced-tiny) subword budget route to the per-pair
    fallback (full text), not the truncating primary path."""
    src, tgt = texts
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="rolling_max", window=9,
                        device=DEVICE, fallback_model=TINY, fallback_layer=TINY_LAYER)
    al._max_subwords = 5  # force overflow routing for any non-trivial text
    out = al.score_batch(src[:6], tgt[:6], return_masks=False)
    assert np.all(np.isfinite(out["bi_score"]))
    # every (non-empty) row overflowed → must equal the fallback per-pair on FULL text
    for i, (s, t) in enumerate(zip(src[:6], tgt[:6])):
        rb, rc = _perpair(tiny_perpair_sa, s, t)
        assert abs(out["bi_score"][i] - rb) < TOL
        assert abs(out["cl_score"][i] - rc) < TOL


def test_edge_cases_finite():
    """Empty, punctuation-only, and long inputs return finite scores, no crash."""
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="rolling_max",
                        window=9, device=DEVICE)
    s = ["", "Привет мир.", "...", " ".join(["слово"] * 50)]
    t = ["Привет мир.", "", "—", " ".join(["текст"] * 50)]
    out = al.score_batch(s, t, return_masks=True)
    assert np.all(np.isfinite(out["bi_score"]))
    assert np.all(np.isfinite(out["cl_score"]))
    assert out["bi_score"][0] == 0.0 and out["cl_score"][0] == 0.0  # empty src


def test_return_shape_matches_alignment_scorer(texts):
    """score_batch return dict has the same keys/shapes as AlignmentScorer."""
    src, tgt = texts
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, device=DEVICE)
    out = al.score_batch(src[:5], tgt[:5], return_masks=True)
    assert set(out) == {"bi_score", "cl_score", "has_bi", "has_cl",
                        "bi_word_mask", "cl_word_mask"}
    assert out["bi_score"].shape == (5,)
    assert out["has_bi"].dtype == bool
    assert len(out["bi_word_mask"]) == 5
    assert "bi_word_mask" not in al.score_batch(src[:5], tgt[:5], return_masks=False)


def test_alias():
    assert BatchedMBertAligner is BatchedAligner


# ── GPU tests (exact large-model parity; skip without CUDA) ─────────────────────
@gpu_only
def test_mbert_l9_rolling_max_parity(golden, texts):
    """Per-row rolling_max bi/cl match the frozen per-pair SimAlign mBERT-L9 golden."""
    src, tgt = texts
    al = BatchedAligner(model="bert", layer=9, score_fn="rolling_max", window=9,
                        device="cuda", enc_batch_size=32)
    out = al.score_batch(src, tgt, return_masks=False)
    g_bi = np.array([r["bi_roll"] for r in golden["scores"]])
    g_cl = np.array([r["cl_roll"] for r in golden["scores"]])
    assert np.abs(out["bi_score"] - g_bi).max() < TOL
    assert np.abs(out["cl_score"] - g_cl).max() < TOL


@gpu_only
def test_mbert_l9_span4_parity(golden, texts):
    src, tgt = texts
    al = BatchedAligner(model="bert", layer=9, score_fn="span", min_span=4,
                        device="cuda", enc_batch_size=32)
    out = al.score_batch(src, tgt, return_masks=False)
    g_bi = np.array([r["bi_span"] for r in golden["scores"]])
    g_cl = np.array([r["cl_span"] for r in golden["scores"]])
    assert np.abs(out["bi_score"] - g_bi).max() < TOL
    assert np.abs(out["cl_score"] - g_cl).max() < TOL


@gpu_only
def test_drop_in_for_alignment_scorer_mbert(texts):
    """Faithful reproduction of AlignmentScorer at its (default) layer 8 (mBERT)."""
    src, tgt = texts
    src, tgt = src[:20], tgt[:20]
    asc = AlignmentScorer(model="bert", method="itermax", score_fn="rolling_max",
                          window=9, device="cuda")  # SimAlign default layer 8
    a = asc.score_batch(src, tgt, return_masks=False)
    del asc; gc.collect(); torch.cuda.empty_cache()
    al8 = BatchedAligner(model="bert", layer=8, score_fn="rolling_max", window=9,
                         device="cuda")
    b = al8.score_batch(src, tgt, return_masks=False)
    assert np.abs(np.array(a["bi_score"]) - b["bi_score"]).max() < TOL
    assert np.abs(np.array(a["cl_score"]) - b["cl_score"]).max() < TOL


@gpu_only
def test_ruroberta_dropin(texts):
    """Production-eval config (ruRoberta layer 8) reproduces AlignmentScorer."""
    src, tgt = texts
    src, tgt = src[:15], tgt[:15]
    asc = AlignmentScorer(model="ai-forever/ruRoberta-large", method="itermax",
                          score_fn="rolling_max", window=9, device="cuda")
    a = asc.score_batch(src, tgt, return_masks=False)
    del asc; gc.collect(); torch.cuda.empty_cache()
    bal = BatchedAligner(model="ai-forever/ruRoberta-large", layer=8,
                         score_fn="rolling_max", window=9, device="cuda", enc_batch_size=8)
    b = bal.score_batch(src, tgt, return_masks=False)
    assert np.abs(np.array(a["bi_score"]) - b["bi_score"]).max() < TOL
    assert np.abs(np.array(a["cl_score"]) - b["cl_score"]).max() < TOL


@gpu_only
def test_ruroberta_production_golden():
    """Production eval config reproduces a COMMITTED frozen reference.

    `golden_ruroberta_l8_bicl.json` freezes per-row bi/cl from
    AlignmentScorer(ruRoberta).score_pair on a fixed eval slice (task 2A.3.1
    step6_benchmark). Unlike test_ruroberta_dropin (a runtime self-cross-check
    needing two model loads), this is a diffable committed contract: the exact
    config wired into eval — ruRoberta L8 / rolling_max w9 + rubert-tiny2 L3
    fallback — must reproduce it <1e-6. Guards against silent drift in the
    production path on any future BatchedAligner change.
    """
    g = json.load(open(GOLDEN_RUROBERTA))
    bal = BatchedAligner(
        model="ai-forever/ruRoberta-large", layer=8, method="itermax",
        score_fn="rolling_max", window=9, device="cuda",
        fallback_model="cointegrated/rubert-tiny2", fallback_layer=3,
        enc_batch_size=8)
    out = bal.score_batch(g["sources"], g["targets"], return_masks=False)
    assert np.abs(np.array(out["bi_score"]) - np.array(g["bi_score"])).max() < TOL
    assert np.abs(np.array(out["cl_score"]) - np.array(g["cl_score"])).max() < TOL
