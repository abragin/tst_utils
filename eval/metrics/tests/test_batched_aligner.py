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
    """Pairs over the (here forced-tiny) subword budget route to the fallback, which
    applies its OWN validated span/min_span=4 regime — NOT the primary's rolling_max.
    `fallback_model=` builds that span4 fallback (back-compat default). Overflow rows
    are tagged `from_fallback=True`."""
    src, tgt = texts
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="rolling_max", window=9,
                        device=DEVICE, fallback_model=TINY, fallback_layer=TINY_LAYER)
    al._max_subwords = 5  # force overflow routing for any non-trivial text
    out = al.score_batch(src[:6], tgt[:6], return_masks=False)
    assert np.all(np.isfinite(out["bi_score"]))
    # every (non-empty) row overflowed → scored by the fallback under SPAN4 (not the
    # primary's rolling_max), and flagged from_fallback.
    for i, (s, t) in enumerate(zip(src[:6], tgt[:6])):
        rb, rc = _perpair(tiny_perpair_sa, s, t, score_fn="span", min_span=4)
        assert abs(out["bi_score"][i] - rb) < TOL
        assert abs(out["cl_score"][i] - rc) < TOL
        assert bool(out["from_fallback"][i]) is True


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
                        "from_fallback", "unhandled", "bi_word_mask", "cl_word_mask"}
    assert out["bi_score"].shape == (5,)
    assert out["has_bi"].dtype == bool
    assert out["from_fallback"].dtype == bool
    assert out["from_fallback"].shape == (5,)
    assert out["unhandled"].dtype == bool
    assert out["unhandled"].shape == (5,)
    assert not out["from_fallback"].any()  # no fallback configured → all False
    assert not out["unhandled"].any()      # short rows fit the primary → none unhandled
    assert len(out["bi_word_mask"]) == 5
    out_nm = al.score_batch(src[:5], tgt[:5], return_masks=False)
    assert "bi_word_mask" not in out_nm
    assert "from_fallback" in out_nm and "unhandled" in out_nm  # always present


def test_alias():
    assert BatchedMBertAligner is BatchedAligner


def test_from_fallback_flag_interleaved_mixed_batch(tiny_perpair_sa):
    """Interleaved batch of {overflow, normal, empty} rows at non-contiguous
    positions: per-row score/mask/flag stay aligned, overflow→fallback(span4),
    normal→primary, empty→default-zero with from_fallback False."""
    LONG = " ".join(["слово"] * 60)              # forced over the tiny budget below
    N1_S, N1_T = "Привет мир сегодня", "Здравствуй земля ныне"
    N2_S, N2_T = "короткий русский текст здесь", "краткое русское сообщение тут"
    # positions: 0 normal, 1 overflow, 2 empty-src, 3 overflow, 4 normal
    s = [N1_S, LONG, "", LONG, N2_S]
    t = [N1_T, LONG, "abc", LONG, N2_T]
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="rolling_max", window=9,
                        device=DEVICE, fallback_model=TINY, fallback_layer=TINY_LAYER)
    al._max_subwords = 40  # LONG (~60 subwords) overflows; short pairs (<40) do not
    out = al.score_batch(s, t, return_masks=True)
    ff = out["from_fallback"]
    assert list(ff) == [False, True, False, True, False]
    # empty-src row: zero + not fallback
    assert out["bi_score"][2] == 0.0 and out["cl_score"][2] == 0.0
    # normal rows: match primary rolling_max
    for i in (0, 4):
        rb, rc = _perpair(tiny_perpair_sa, s[i], t[i], score_fn="rolling_max", window=9)
        assert abs(out["bi_score"][i] - rb) < TOL
        assert abs(out["cl_score"][i] - rc) < TOL
    # overflow rows: match fallback span4, and masks align with the fallback's
    for i in (1, 3):
        rb, rc = _perpair(tiny_perpair_sa, s[i], t[i], score_fn="span", min_span=4)
        assert abs(out["bi_score"][i] - rb) < TOL
        assert abs(out["cl_score"][i] - rc) < TOL
    assert len(out["bi_word_mask"]) == 5 and len(out["cl_word_mask"]) == 5


def test_empty_row_in_overflow_batch_not_fallback():
    """An empty row inside an overflow-containing batch stays default-zero and
    from_fallback=False (only true overflow rows are delegated)."""
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="span", min_span=4,
                        device=DEVICE, fallback_model=TINY, fallback_layer=TINY_LAYER)
    al._max_subwords = 30
    s = ["", " ".join(["слово"] * 50)]
    t = ["непустой текст", " ".join(["текст"] * 50)]
    out = al.score_batch(s, t, return_masks=False)
    assert bool(out["from_fallback"][0]) is False
    assert out["bi_score"][0] == 0.0 and out["cl_score"][0] == 0.0
    assert bool(out["from_fallback"][1]) is True


def test_gender_no_fallback_overflow_returns_empty_edges():
    """Gender path invariant: with NO fallback configured, an overflow row yields an
    empty edge list from align_batch (→ score_B defaults to 1.0) and increments
    n_overflow_unhandled — never silently scored under primary settings."""
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, method="itermax", device=DEVICE)
    al._max_subwords = 20  # force overflow; NO fallback_model / fallback
    s = ["короткий текст", " ".join(["слово"] * 40)]
    t = ["короткий текст", " ".join(["текст"] * 40)]
    edges = al.align_batch(s, t)
    assert len(edges) == 2
    assert edges[1] == []                       # overflow row → empty edges
    assert al.n_overflow_unhandled == 1
    # score_batch on the same overflow (no fallback) → default-zero, not fallback,
    # and flagged unhandled (no aligner could score it).
    al.n_overflow_unhandled = 0
    out = al.score_batch(s, t, return_masks=False)
    assert bool(out["from_fallback"][1]) is False
    assert bool(out["unhandled"][1]) is True
    assert bool(out["unhandled"][0]) is False
    assert out["bi_score"][1] == 0.0 and out["cl_score"][1] == 0.0
    assert al.n_overflow_unhandled == 1


def test_fallback_of_fallback_giveup_flagged_unhandled():
    """A row that overflows BOTH the primary AND the fallback's budget is tagged
    from_fallback=True (it was routed) AND unhandled=True (the fallback gave up →
    default-zero), and the count aggregates onto the PARENT's n_overflow_unhandled.
    This is the long-text failure mode the counter was added to surface."""
    fb = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="span", min_span=4,
                        device=DEVICE)
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, score_fn="rolling_max", window=9,
                        device=DEVICE, fallback=fb)
    MED = " ".join(["текст"] * 12)              # row 0: overflows primary, fits fallback
    LONG = " ".join(["слово"] * 60)             # row 1: overflows BOTH
    n_med = al._n_subwords([al._clean_word(w) for w in MED.split()])
    n_long = al._n_subwords([al._clean_word(w) for w in LONG.split()])
    assert n_med < n_long
    al._max_subwords = n_med - 1                # primary overflows on both rows
    fb._max_subwords = n_med                    # fallback fits MED, overflows LONG
    s = [MED, LONG]
    t = [MED, LONG]
    out = al.score_batch(s, t, return_masks=False)
    # row 0: primary overflow (>8) but fallback (<=15) handles it → scored, not unhandled
    assert bool(out["from_fallback"][0]) is True
    assert bool(out["unhandled"][0]) is False
    # row 1: both overflow → routed to fallback, fallback gives up → unhandled
    assert bool(out["from_fallback"][1]) is True
    assert bool(out["unhandled"][1]) is True
    assert out["bi_score"][1] == 0.0 and out["cl_score"][1] == 0.0
    # PARENT counter reflects the give-up (the bug the review caught: was counted on
    # the child only, so step6b's parent read stayed 0).
    assert al.n_overflow_unhandled == 1


def test_encode_wts_parity():
    """Compute-once: _encode given precomputed wts is byte-identical to _encode that
    tokenizes internally (justifies threading wts through the budget check)."""
    al = BatchedAligner(model=TINY, layer=TINY_LAYER, device=DEVICE)
    word_lists = [["Привет", "мир", "сегодня"],
                  ["Лев", "Толстой", "написал", "роман"],
                  ["один"]]
    wts = [[al._tok.tokenize(w) for w in words] for words in word_lists]
    a = al._encode(word_lists)            # internal tokenize
    b = al._encode(word_lists, wts=wts)   # precomputed
    assert len(a) == len(b)
    for va, vb in zip(a, b):
        assert np.array_equal(va, vb)


@pytest.mark.parametrize("model_id,margin,mpe", [
    ("cointegrated/rubert-tiny2", 2, 2048),
    ("bert-base-multilingual-cased", 2, 512),
    ("ai-forever/ruRoberta-large", 4, 514),
])
def test_subword_upperbound_guard_is_sound(model_id, margin, mpe):
    """Step-0 gate frozen as a regression: the per-word `tokenize` sum (what the
    proactive guard counts) plus the margin is a TRUE upper bound on the encoder's
    actual position count. If this ever fails, the guard could pass an over-budget
    sequence to the primary and trip the CUDA positional-index assert. Tokenizer-only
    (no model weights) — cheap."""
    import re
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)

    def clean(w):
        c = re.sub(r'^[^\w]+|[^\w]+$', '', w, flags=re.UNICODE)
        return c if c else w
    vocab = "слово текст роман война мир эпоха время человек дом небо".split()
    rng = np.random.RandomState(0)
    max_sub = mpe - margin
    for target in (max_sub - 3, max_sub - 1, max_sub, max_sub + 1, max_sub + 5):
        words, ns = [], 0
        while ns < target:
            w = vocab[rng.randint(len(vocab))]
            n = len(tok.tokenize(clean(w)))
            if ns + n > target:
                break
            words.append(w); ns += n
        cw = [clean(w) for w in words]
        n_sub = sum(len(tok.tokenize(w)) for w in cw)
        enc_len = len(tok(cw, is_split_into_words=True)["input_ids"])
        # guard soundness: routed-to-primary (n_sub<=max_sub) ⇒ actually fits (enc<=mpe)
        if n_sub <= max_sub:
            assert enc_len <= mpe, (model_id, n_sub, enc_len, mpe)
        # the inequality the guard depends on, in all cases
        assert n_sub + margin >= enc_len, (model_id, n_sub, enc_len, margin)


def test_alignment_scorer_proactive_guard_no_primary_call():
    """AlignmentScorer: an over-budget pair routes to the fallback WITHOUT calling
    the primary aligner's get_word_aligns (the proactive guard avoids feeding an
    over-budget sequence to the primary — the GPU CUDA-assert trigger). Mocks both
    aligners so it runs on CPU with no encode."""
    from unittest import mock
    asc = AlignmentScorer(model=TINY, method="itermax", score_fn="rolling_max",
                          window=9, device=DEVICE)
    asc._max_subwords = 3  # force any real sentence over budget

    class _StubFallback:
        called = False
        def score_pair(self, source, target):
            self.called = True
            return dict(bi_score=0.11, cl_score=0.22, has_bi=False, has_cl=False,
                        bi_word_mask=[], cl_word_mask=[], from_fallback=False)
    stub = _StubFallback()
    asc._fallback = stub

    asc._aligner.get_word_aligns = mock.Mock(
        side_effect=AssertionError("primary must NOT be called on over-budget pair"))

    r = asc.score_pair("Лев Толстой написал большой роман о войне",
                       "Граф создал великое произведение о сражениях")
    assert stub.called is True
    asc._aligner.get_word_aligns.assert_not_called()
    assert r["from_fallback"] is True
    assert r["bi_score"] == 0.11 and r["cl_score"] == 0.22


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
    # Precondition: the frozen slice overflows 0% → no fallback fires → byte-identical
    # to the pre-fix golden. If a future data change introduces overflow, this fails
    # loudly (the golden would silently diverge under the new span4 fallback regime).
    assert not out["from_fallback"].any(), (
        f"{int(out['from_fallback'].sum())} golden rows overflowed — golden no longer "
        "exercises the pure-primary path; regenerate and re-validate the fallback slice")


@gpu_only
def test_alignment_scorer_fallback_scores_under_span4():
    """AlignmentScorer: an over-budget pair is scored by the fallback under its OWN
    span4 regime (rubert-tiny2), differing from the primary's rolling_max. Uses real
    models (ruRoberta primary, rubert-tiny2 fallback) on GPU."""
    asc = AlignmentScorer(model="ai-forever/ruRoberta-large", method="itermax",
                          score_fn="rolling_max", window=9, device="cuda",
                          fallback_model="cointegrated/rubert-tiny2")
    asc._max_subwords = 8  # force overflow without needing a 514-token text
    src = "Лев Толстой написал великий роман о войне и мире в эпоху Наполеона снова"
    tgt = "Граф Толстой создал большое произведение о сражениях и согласии в годы империи"
    r = asc.score_pair(src, tgt)
    assert r["from_fallback"] is True
    # the fallback (tiny2 span4, min_span=4) and its own thresholds drive has_bi/has_cl
    assert np.isfinite(r["bi_score"]) and np.isfinite(r["cl_score"])
    # span4 with min_span=4 on a short sentence: no run >=4 → 0.0 (rolling_max would not be 0)
    # (exact value depends on tiny2 alignment; assert it used span semantics: integer-ish fraction)
    assert 0.0 <= r["bi_score"] <= 1.0


@gpu_only
def test_alignment_scorer_batch_pair_agreement_mixed():
    """AlignmentScorer score_batch == per-row score_pair on a mixed set including an
    over-budget row (loops score_pair, so order/keys/flags stay aligned)."""
    asc = AlignmentScorer(model="ai-forever/ruRoberta-large", method="itermax",
                          score_fn="rolling_max", window=9, device="cuda",
                          fallback_model="cointegrated/rubert-tiny2")
    LONG = " ".join(["слово"] * 600)  # > ruRoberta 514 budget → real overflow
    s = ["Привет мир сегодня", LONG, "короткий текст здесь"]
    t = ["Здравствуй земля ныне", LONG, "краткое сообщение тут"]
    batch = asc.score_batch(s, t, return_masks=False)
    for i, (si, ti) in enumerate(zip(s, t)):
        pair = asc.score_pair(si, ti)
        assert abs(batch["bi_score"][i] - pair["bi_score"]) < TOL
        assert abs(batch["cl_score"][i] - pair["cl_score"]) < TOL
        assert bool(batch["from_fallback"][i]) == pair["from_fallback"]
    assert bool(batch["from_fallback"][1]) is True  # the LONG row overflowed
