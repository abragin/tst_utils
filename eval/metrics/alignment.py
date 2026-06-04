"""
BI/CL detection via word alignment (SimAlign).

Boilerplate Injection (BI): target words with no alignment to source.
Content Loss (CL): source words with no alignment to target.

Two scorers (same return contract):
  - `AlignmentScorer`     — per-pair SimAlign; the original/reference implementation.
  - `BatchedAligner`      — batched, source-cached, model-agnostic drop-in (task
                            2A.3.1). Batches the encode and reuses each source
                            across its N targets; reproduces AlignmentScorer
                            bit-exactly at matched layer. Use ruRoberta for eval
                            BI/CL (zero drift, ~1.8×), mBERT for filtering/gender.
                            See the class docstring for the rationale.

Usage:
    scorer = AlignmentScorer()
    results = scorer.score_batch(sources, targets)
    # results: dict with bi_score, cl_score, bi_word_mask, cl_word_mask

Validated on 500-sample 1.4b LLM-labeled set (ruRoberta-large, itermax):
    score_fn='rolling_max', window=9:  BI F1=0.735, AUC=0.907  ← best overall
    score_fn='span',        min_span=4: BI F1=0.689, P=0.702   ← best precision (data filtering)

Model comparison (itermax, punctuation stripping):
    bert-multilingual:  span4 F1=0.689, P=0.702 ← best precision, lower cost
    rubert-tiny2:       span4 F1=0.707, AUC=0.841, 6ms/pair   ← speed fallback
    ruBert-base:        span4 F1=0.669, AUC=0.822, 11ms/pair
    ruRoberta-large:    rolling_max F1=0.735, AUC=0.907, 23ms/pair ← best overall
                        ⚠ Do NOT use on texts with gender morphology changes (написал/написала) —
                          contextual embeddings flag inflected cognates as CL, giving 17/34 false
                          CL positives on gender-switch validation data. Use bert-multilingual instead.

Scoring functions:
    score_fn='rolling_max' (default):
        Per-word local unalignment density via uniform convolution window; bi_score =
        max(density). Handles BI spans broken by 1-2 accidentally-aligned words;
        per-word density map useful for BI region removal in post-processing.
        Recommended threshold: bi_threshold=0.75.
    score_fn='span':
        Count only unaligned words in contiguous runs >= min_span.
        Best precision for data filtering; simpler to interpret.
        Recommended threshold: bi_threshold=0.15 (min_span=4).

Punctuation stripping (_clean_word) is applied before alignment to prevent e.g.
'справочник' vs 'справочник...' from failing to align.
"""

import logging
import re
from typing import List

import numpy as np

LOG = logging.getLogger(__name__)


class AlignmentScorer:
    """
    Word-alignment-based BI/CL scorer using SimAlign.

    Parameters
    ----------
    model : str
        SimAlign model alias. "bert" = bert-base-multilingual-cased (cached on tallin.vpn).
        "xlmr" = xlm-roberta-base. Russian models: "cointegrated/rubert-tiny2",
        "ai-forever/ruBert-base", "ai-forever/ruRoberta-large".
    method : str
        Alignment method: "itermax" (recommended), "inter", "fwd", "rev".
    bi_threshold : float or None
        Threshold above which a pair is flagged as BI. None = use default for score_fn:
        0.75 for 'rolling_max', 0.25 for 'span'.
    cl_threshold : float
        Threshold above which a pair is flagged as CL (raw fraction, no filter applied).
    score_fn : str
        'rolling_max' (default): peak local density via uniform convolution window.
            Better F1/AUC overall; recommended for detection and BI removal.
        'span': contiguous span filter. Best precision; recommended for data filtering.
    window : int
        (score_fn='rolling_max') Kernel width for local density. Default: 9.
    min_span : int
        (score_fn='span') Min contiguous unaligned run length to count.
        1 = raw fraction. Recommended: 2 for metrics, 4 for data filtering.
    device : str
        "cuda" or "cpu".
    fallback : AlignmentScorer or None
        Preconfigured scorer used for pairs whose source or target exceeds the
        primary model's positional-embedding budget. The fallback applies its OWN
        model + score_fn/window/min_span/thresholds (e.g. rubert-tiny2 with its
        validated span/min_span=4 regime), not the primary's. Mutually exclusive
        with `fallback_model`. Overflow rows are detected proactively (subword
        pre-count, no primary forward pass) so the over-budget sequence never
        reaches the primary encoder.
    fallback_model : str or None
        Back-compat shorthand: builds an internal fallback `AlignmentScorer` on this
        model under the validated span/min_span=4 regime. Recommended:
        "cointegrated/rubert-tiny2" (max_position_embeddings=2048, span4 F1=0.707).
        bert-base-multilingual-cased has the same 512 limit and is NOT a useful
        fallback. Mutually exclusive with `fallback`.
    max_words_per_side : int or None
        If set, truncate source and target to this many words before alignment.
        Applied BEFORE the primary model; use as a last resort when even the
        fallback model cannot handle the text length.
    """

    DEFAULT_CL_THRESHOLD = 0.25

    def __init__(
        self,
        model: str = "bert",
        method: str = "itermax",
        bi_threshold: float = None,
        cl_threshold: float = DEFAULT_CL_THRESHOLD,
        score_fn: str = "rolling_max",
        window: int = 9,
        min_span: int = 1,
        device: str = "cuda",
        max_words_per_side: int = None,
        fallback_model: str = None,
        fallback: "AlignmentScorer" = None,
    ):
        try:
            from simalign import SentenceAligner
        except ImportError:
            raise ImportError("simalign is required: pip install simalign")

        if fallback is not None and fallback_model is not None:
            raise ValueError(
                "pass either `fallback` (a preconfigured AlignmentScorer) or "
                "`fallback_model` (str), not both")

        self.method = method
        self.bi_threshold = bi_threshold if bi_threshold is not None else (
            0.75 if score_fn == "rolling_max" else 0.25
        )
        self.cl_threshold = cl_threshold
        self.score_fn = score_fn
        self.window = window
        self.min_span = min_span
        self.max_words_per_side = max_words_per_side

        def _make_aligner(model_id):
            al = SentenceAligner(
                model=model_id,
                token_type="word",
                matching_methods=self._method_char(method),
                device=device,
            )
            # RoBERTa-family tokenizers require add_prefix_space=True for pretokenized
            # inputs (simalign calls tokenizer with is_split_into_words=True).
            tok = al.embed_loader.tokenizer
            if getattr(tok, "add_prefix_space", None) is False:
                from transformers import AutoTokenizer
                al.embed_loader.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, add_prefix_space=True
                )
            # Clamp the requested layer to the model depth: SimAlign defaults to
            # layer 8, but small fallback models (e.g. rubert-tiny2, 3 hidden layers)
            # have fewer — without this the fallback raised ValueError at encode time
            # (the latent reason the tiny2 fallback never actually worked). No-op for
            # deep models (ruRoberta 24, mBERT 12). tiny2 → layer 3 (validated).
            n_hidden = al.embed_loader.emb_model.config.num_hidden_layers
            if al.embed_loader.layer > n_hidden:
                al.embed_loader.layer = n_hidden
            return al

        self._aligner = _make_aligner(model)
        self._tok = self._aligner.embed_loader.tokenizer

        # Proactive over-budget guard: a pair whose per-word subword sum exceeds this
        # budget is routed to the fallback BEFORE the primary forward pass, so the
        # over-budget sequence never trips the primary encoder's positional-index
        # assert. The per-word `tokenize` sum is a verified true upper bound on the
        # encoder's content-position count (task 2A.3.2 Step 0); the margin covers
        # the 2 special tokens. RoBERTa-family reserve extra positions (padding_idx).
        cfg = self._aligner.embed_loader.emb_model.config
        mpe = getattr(cfg, "max_position_embeddings", 512)
        mt = getattr(cfg, "model_type", "")
        margin = 4 if "roberta" in str(mt).lower() else 2
        self._max_subwords = mpe - margin

        # Fallback as a preconfigured scorer that applies its OWN scoring regime.
        if fallback is not None:
            self._fallback = fallback
        elif fallback_model is not None:
            # Back-compat: build a span4 fallback under the fallback model's own
            # validated regime (was: inherit the primary's score_fn/window — the bug).
            self._fallback = AlignmentScorer(
                model=fallback_model, method=method, score_fn="span", min_span=4,
                cl_threshold=cl_threshold, device=device,
                max_words_per_side=max_words_per_side,
            )
        else:
            self._fallback = None

    def _compute_score(self, unaligned_indices: list, total: int) -> float:
        """Dispatch to the configured scoring function."""
        if self.score_fn == "rolling_max":
            return self._rolling_density_score(unaligned_indices, total, self.window)
        return self._span_score(unaligned_indices, total)

    @staticmethod
    def rolling_density_map(unaligned_indices: list, total: int, window: int = 9) -> np.ndarray:
        """Per-word unalignment density array (uniform kernel, edge-normalised).

        density[i] = fraction of words in the window centred on i that are unaligned,
        normalised by actual in-bounds window size so edge positions are not penalised.
        Returns an array of length `total` with values in [0, 1].
        """
        if total == 0:
            return np.zeros(0)
        unaligned = np.zeros(total)
        for i in unaligned_indices:
            if 0 <= i < total:
                unaligned[i] = 1.0
        k = np.ones(window) / window
        norm = np.convolve(np.ones(total), k, mode="same")
        return (np.convolve(unaligned, k, mode="same") / norm)[:total]

    @staticmethod
    def _rolling_density_score(unaligned_indices: list, total: int, window: int = 9) -> float:
        """Peak local unalignment density (uniform kernel, edge-normalised).

        density[i] = fraction of words in the window centred on i that are unaligned,
        normalised by actual in-bounds window size so edge positions are not penalised.
        bi_score = max(density) over all positions.
        """
        if total == 0 or not unaligned_indices:
            return 0.0
        density = AlignmentScorer.rolling_density_map(unaligned_indices, total, window)
        return float(np.max(density))

    def _span_score(self, unaligned_indices: list, total: int) -> float:
        """Fraction of words in contiguous unaligned runs of length >= min_span."""
        if not unaligned_indices or total == 0:
            return 0.0
        if self.min_span <= 1:
            return len(unaligned_indices) / total
        indices = sorted(unaligned_indices)
        span_count = 0
        run_start = 0
        for k in range(1, len(indices) + 1):
            if k == len(indices) or indices[k] != indices[k - 1] + 1:
                if k - run_start >= self.min_span:
                    span_count += k - run_start
                run_start = k
        return span_count / total

    @staticmethod
    def _method_char(method: str) -> str:
        mapping = {"itermax": "i", "inter": "a", "mwmf": "m", "fwd": "f", "rev": "r"}
        if method not in mapping:
            raise ValueError(f"Unknown alignment method: {method}. Choose from {list(mapping)}")
        return mapping[method]

    @staticmethod
    def _clean_word(w: str) -> str:
        """Strip leading/trailing punctuation before passing to aligner.

        Prevents tokens like 'справочник...' from failing to match 'справочник'.
        Falls back to the original if the word is entirely punctuation (e.g. '—').
        """
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', w, flags=re.UNICODE)
        return cleaned if cleaned else w

    def _n_subwords(self, clean_words: List[str]) -> int:
        return sum(len(self._tok.tokenize(w)) for w in clean_words)

    def _primary_edges(self, src_clean: List[str], tgt_clean: List[str]):
        """Primary-model alignment only. May raise RuntimeError if the sequence
        exceeds the encoder's positional budget (caught by the reactive guard)."""
        al = self._aligner.get_word_aligns(src_clean, tgt_clean)[self.method]
        return {i for i, j in al}, {j for i, j in al}

    def score_pair(self, source: str, target: str) -> dict:
        """
        Score a single (source, target) pair.

        Returns
        -------
        dict with:
            bi_score      : float — BI score (rolling_max peak or span fraction)
            cl_score      : float — fraction of source words unaligned to target (raw)
            has_bi        : bool  — bi_score >= bi_threshold
            has_cl        : bool  — cl_score >= cl_threshold
            bi_word_mask  : list[tuple[int, str]] — (index, word) for unaligned target words
            cl_word_mask  : list[tuple[int, str]] — (index, word) for unaligned source words
            from_fallback : bool  — True if this pair was scored by the fallback scorer
                            (its own model + score regime), always present (False when
                            no fallback configured or the pair fit the primary budget).
            unhandled     : bool  — True if no aligner in the chain could score this
                            pair (over budget with no fallback, or the primary failed
                            unexpectedly with no fallback). Carries default-zero scores
                            → optimistic bi_cl; downstream can flag/exclude. Always present.
        """
        src_words = source.split()
        tgt_words = target.split()

        if self.max_words_per_side is not None:
            src_words = src_words[:self.max_words_per_side]
            tgt_words = tgt_words[:self.max_words_per_side]

        if not src_words or not tgt_words:
            return dict(bi_score=0.0, cl_score=0.0, has_bi=False, has_cl=False,
                        bi_word_mask=[], cl_word_mask=[], from_fallback=False,
                        unhandled=False)

        src_clean = [self._clean_word(w) for w in src_words]
        tgt_clean = [self._clean_word(w) for w in tgt_words]

        # Proactive over-budget guard: route to the fallback BEFORE touching the
        # primary encoder (the per-word subword sum is a verified upper bound on the
        # encoder's position count — task 2A.3.2 Step 0), so an over-budget sequence
        # never trips the primary's positional-index assert.
        over_budget = (self._n_subwords(src_clean) > self._max_subwords
                       or self._n_subwords(tgt_clean) > self._max_subwords)
        if over_budget and self._fallback is not None:
            r = self._fallback.score_pair(source, target)
            r["from_fallback"] = True
            return r

        unhandled = False
        try:
            aligned_src, aligned_tgt = self._primary_edges(src_clean, tgt_clean)
        except RuntimeError:
            # Reactive last resort: the guard should have caught all length overflow,
            # so this fires only on unexpected encoder failures. Delegate the whole
            # pair to the fallback's own scoring regime when available.
            if self._fallback is not None:
                r = self._fallback.score_pair(source, target)
                r["from_fallback"] = True
                return r
            # over budget (or other failure) with no fallback → unscored, default-zero
            aligned_src, aligned_tgt = set(), set()
            unhandled = True

        bi_mask = [(j, tgt_words[j]) for j in range(len(tgt_words)) if j not in aligned_tgt]
        cl_mask = [(i, src_words[i]) for i in range(len(src_words)) if i not in aligned_src]

        bi_score = self._compute_score([j for j, _ in bi_mask], len(tgt_words))
        cl_score = self._compute_score([i for i, _ in cl_mask], len(src_words))

        return dict(
            bi_score=bi_score,
            cl_score=cl_score,
            has_bi=bi_score >= self.bi_threshold,
            has_cl=cl_score >= self.cl_threshold,
            bi_word_mask=bi_mask,
            cl_word_mask=cl_mask,
            from_fallback=False,
            unhandled=unhandled,
        )

    def score_batch(
        self,
        sources: List[str],
        targets: List[str],
        return_masks: bool = True,
    ) -> dict:
        """
        Score a batch of (source, target) pairs.

        Parameters
        ----------
        sources, targets : list of str
        return_masks : bool
            If False, skip per-word mask lists (faster, lower memory for large batches).

        Returns
        -------
        dict with numpy arrays / lists:
            bi_score      : np.ndarray[float]
            cl_score      : np.ndarray[float]
            has_bi        : np.ndarray[bool]
            has_cl        : np.ndarray[bool]
            from_fallback : np.ndarray[bool] — True for rows scored by the fallback
            unhandled     : np.ndarray[bool] — True for rows no aligner could score
            bi_word_mask  : list[list[tuple]] — only if return_masks=True
            cl_word_mask  : list[list[tuple]] — only if return_masks=True
        """
        bi_scores, cl_scores, from_fb, unhandled = [], [], [], []
        bi_masks, cl_masks = [], []

        for src, tgt in zip(sources, targets):
            r = self.score_pair(src, tgt)
            bi_scores.append(r["bi_score"])
            cl_scores.append(r["cl_score"])
            from_fb.append(r["from_fallback"])
            unhandled.append(r["unhandled"])
            if return_masks:
                bi_masks.append(r["bi_word_mask"])
                cl_masks.append(r["cl_word_mask"])

        out = dict(
            bi_score=np.array(bi_scores),
            cl_score=np.array(cl_scores),
            has_bi=np.array(bi_scores) >= self.bi_threshold,
            has_cl=np.array(cl_scores) >= self.cl_threshold,
            from_fallback=np.array(from_fb, dtype=bool),
            unhandled=np.array(unhandled, dtype=bool),
        )
        if return_masks:
            out["bi_word_mask"] = bi_masks
            out["cl_word_mask"] = cl_masks
        return out


class BatchedAligner:
    """Batched, source-cached, **model-agnostic** drop-in for `AlignmentScorer`.

    Background (task 2A.3.1): the per-pair `AlignmentScorer` re-encodes the source
    on every pair, and SimAlign couples encoding with matching. But a text's
    layer-L word-vectors are independent of its batch partner (attention is masked
    per-sentence — verified bit-exact in the Step-2 spike for both mBERT and
    RoBERTa-family models), and `iter_max` is cheap argmax matching on the
    cosine-similarity matrix. So we:

      1. group rows by source string,
      2. batch-encode each source's tokens **once** + that group's targets,
      3. run itermax per (source, target) on the cached word-vectors,
      4. drop the group's token embeddings before the next group (transient).

    This captures the 1-source→N-targets reuse (≈22× in eval) that is moot for
    BERTScore meaning (the library dedups internally) but wide open for alignment.
    Memory is bounded by one source-group of word-vectors, not the dataset.

    **Model-agnostic** (task-2A.3.1 Step-4 decision): the speedup is in batching
    the per-pair loop, not the encoder. Configure `model` per use:
      - **eval BI/CL**: `model="ai-forever/ruRoberta-large", layer=8,
        fallback_model="cointegrated/rubert-tiny2"` — reproduces the production
        per-pair scores exactly (zero drift), keeps the long-text fallback.
      - **gender / 2A.4 filtering**: `model="bert"` (mBERT). mBERT L9 was validated
        for BI/CL too (AUC 0.900 vs ruRoberta 0.907) but is more lenient — see the
        drift analysis; not used for the production eval BI/CL.

    `fallback_model` mirrors `AlignmentScorer`: pairs whose source or target exceeds
    the primary model's subword budget are scored per-pair with the fallback
    (cold path — 0% of the short eval set). Reuses SimAlign's tokenizer, embedding
    model, and static `get_similarity`/`apply_distortion`/`iter_max`.
    """

    DEFAULT_CL_THRESHOLD = 0.25

    def __init__(
        self,
        model: str = "bert",
        layer: int = 8,
        method: str = "itermax",
        bi_threshold: float = None,
        cl_threshold: float = DEFAULT_CL_THRESHOLD,
        score_fn: str = "rolling_max",
        window: int = 9,
        min_span: int = 1,  # matches AlignmentScorer; use 4 for span-based data filtering
        device: str = "cuda",
        enc_batch_size: int = 32,
        max_words_per_side: int = None,
        distortion: float = 0.0,
        fallback_model: str = None,
        fallback_layer: int = None,
        fallback: "BatchedAligner" = None,
    ):
        try:
            from simalign import SentenceAligner
        except ImportError:
            raise ImportError("simalign is required: pip install simalign")

        if method != "itermax":
            raise NotImplementedError("BatchedAligner currently supports itermax only")

        if fallback is not None and fallback_model is not None:
            raise ValueError(
                "pass either `fallback` (a preconfigured BatchedAligner) or "
                "`fallback_model` (str), not both")

        self.method = method
        self.layer = layer
        # Default threshold matches AlignmentScorer (0.75 rolling_max / 0.25 span);
        # callers override (e.g. mBERT recalibrated to 0.80).
        self.bi_threshold = bi_threshold if bi_threshold is not None else (
            0.75 if score_fn == "rolling_max" else 0.25)
        self.cl_threshold = cl_threshold
        self.score_fn = score_fn
        self.window = window
        self.min_span = min_span
        self.enc_batch_size = enc_batch_size
        self.max_words_per_side = max_words_per_side
        self.distortion = distortion
        self._device_str = device

        # Reuse SimAlign's tokenizer + emb model + static matching (bit-exact, Step-2 spike).
        self._sa = SentenceAligner(
            model=model, token_type="word", matching_methods="i",
            device=device, layer=layer,
        )
        self._tok = self._sa.embed_loader.tokenizer
        self._emb_model = self._sa.embed_loader.emb_model
        self._device = self._sa.embed_loader.device
        # RoBERTa-family tokenizers need add_prefix_space for pretokenized input.
        if getattr(self._tok, "add_prefix_space", None) is False:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(
                self._sa.model, add_prefix_space=True
            )
            self._sa.embed_loader.tokenizer = self._tok

        # Subword budget for the primary model (overflow → fallback). RoBERTa-family
        # reserve extra positions (padding_idx offset), so subtract a margin.
        cfg = self._emb_model.config
        mpe = getattr(cfg, "max_position_embeddings", 512)
        mt = getattr(cfg, "model_type", "")
        margin = 4 if "roberta" in str(mt).lower() else 2
        self._max_subwords = mpe - margin

        # Fallback as a preconfigured BatchedAligner that applies its OWN model +
        # score regime (e.g. rubert-tiny2 span/min_span=4). Overflow rows are routed
        # here proactively (subword pre-count), so the over-budget sequence never
        # reaches the primary encoder. A `fallback=` instance is eager; `fallback_model=`
        # is built lazily on first overflow (cold path — 0% of the short eval set).
        self._fallback = fallback
        self._fallback_model = fallback_model
        self._fallback_layer = fallback_layer
        # Count of overflow rows this aligner could NOT route to a fallback (no
        # fallback configured) — surfaces fallback-of-fallback failures instead of
        # silently zeroing. Gender (no fallback) increments this by design.
        self.n_overflow_unhandled = 0

    # ── scoring helpers (mirror AlignmentScorer) ───────────────────────────────
    @staticmethod
    def rolling_density_map(unaligned_indices, total, window=9):
        return AlignmentScorer.rolling_density_map(unaligned_indices, total, window)

    def _compute_score(self, unaligned_indices, total):
        if self.score_fn == "rolling_max":
            return AlignmentScorer._rolling_density_score(unaligned_indices, total, self.window)
        # span
        if not unaligned_indices or total == 0:
            return 0.0
        if self.min_span <= 1:
            return len(unaligned_indices) / total
        idx = sorted(unaligned_indices)
        cnt, run = 0, 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or idx[k] != idx[k - 1] + 1:
                if k - run >= self.min_span:
                    cnt += k - run
                run = k
        return cnt / total

    # ── encoding ────────────────────────────────────────────────────────────────
    def _avg_words(self, subword_vecs, word_tokens):
        """Average subword vectors → per-word vectors (SimAlign semantics)."""
        out, cnt = [], 0
        h = subword_vecs.shape[1] if subword_vecs.ndim == 2 else 0
        for wt in word_tokens:
            n = len(wt)
            if n == 0 or cnt >= subword_vecs.shape[0]:
                out.append(np.zeros(h, dtype=np.float32))
            else:
                end = min(cnt + n, subword_vecs.shape[0])
                out.append(subword_vecs[cnt:end].mean(0))
            cnt += n
        return np.asarray(out, dtype=np.float32)

    def _encode(self, word_lists, wts=None):
        """Batched encode of pretokenized texts → list of per-word vector arrays.

        Each text's vectors are independent of batch partners (masked attention),
        so this is safe to call per source-group. Assumes inputs have already
        passed the subword-budget check in `_edges_per_row` (overflow rows are
        routed to the fallback before reaching here); the `truncation=True`
        cap is only a defensive guard, not the long-text handling path.

        `wts` (per-word subword-token lists) may be passed in to avoid re-tokenizing
        words already tokenized for the budget check; when None it is computed here
        (byte-identical — covered by a parity test).
        """
        import torch
        out = [None] * len(word_lists)
        if wts is None:
            wts = [[self._tok.tokenize(w) for w in words] for words in word_lists]
        for s in range(0, len(word_lists), self.enc_batch_size):
            chunk = word_lists[s:s + self.enc_batch_size]
            with torch.no_grad():
                inp = self._tok(chunk, is_split_into_words=True, padding=True,
                                truncation=True, return_tensors="pt").to(self._device)
                hidden = self._emb_model(**inp)["hidden_states"]
                if self.layer >= len(hidden):
                    raise ValueError(
                        f"layer {self.layer} but model has {len(hidden)} hidden states")
                hidden = hidden[self.layer][:, 1:-1, :].cpu().numpy()
            for k in range(len(chunk)):
                wt = wts[s + k]
                n_bpe = sum(len(w) for w in wt)
                # cap to actually-encoded width (truncation may have dropped tail subwords)
                avail = hidden[k].shape[0]
                out[s + k] = self._avg_words(hidden[k, :min(n_bpe, avail)], wt)
        return out

    def _edges_from_vecs(self, v_src, v_tgt):
        """Aligned word-index edges (i, j) from precomputed word-vectors."""
        sim = self._sa.get_similarity(v_src, v_tgt)
        sim = self._sa.apply_distortion(sim, self.distortion)
        mat = self._sa.iter_max(sim)
        return [(int(i), int(j)) for i, j in np.argwhere(mat > 0)]

    @staticmethod
    def _clean_word(w: str) -> str:
        """Strip leading/trailing punctuation before alignment (matches AlignmentScorer)."""
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', w, flags=re.UNICODE)
        return cleaned if cleaned else w

    def _words(self, text):
        w = text.split() if isinstance(text, str) else list(text)
        if self.max_words_per_side is not None:
            w = w[:self.max_words_per_side]
        return w

    def _tokenize_words(self, clean_words):
        """Per-word subword-token lists (computed once; reused for budget + encode)."""
        return [self._tok.tokenize(w) for w in clean_words]

    def _n_subwords(self, clean_words):
        return sum(len(t) for t in self._tokenize_words(clean_words))

    def _ensure_fallback(self):
        """Lazily build the back-compat `fallback_model` aligner under the validated
        span/min_span=4 regime (was: inherit the primary's score_fn — the bug). A
        `fallback=` instance is already built and skips this path."""
        if self._fallback is not None or not self._fallback_model:
            return
        req = self._fallback_layer if self._fallback_layer is not None else self.layer
        fb = BatchedAligner(
            model=self._fallback_model, layer=req, method="itermax",
            score_fn="span", min_span=4, cl_threshold=self.cl_threshold,
            device=self._device_str, enc_batch_size=self.enc_batch_size,
            max_words_per_side=self.max_words_per_side, distortion=self.distortion,
        )
        # clamp layer for small fallbacks (e.g. rubert-tiny2 has 3 hidden layers)
        n_hidden = fb._emb_model.config.num_hidden_layers
        if fb.layer > n_hidden:
            fb.layer = n_hidden
        self._fallback = fb

    def _edges_per_row(self, sources, targets):
        """Order-preserving per-row alignment with source-group reuse.

        Returns ``(rows, overflow_idx)`` where:
          - ``rows[i] = (edges, src_words, tgt_words)``; ``edges`` is the list of
            aligned (i, j) word-index pairs for primary-scored rows, or **None** for
            rows left at default-zero — *both* empty/degenerate rows AND overflow
            rows (overflow is no longer scored under the primary's settings here).
          - ``overflow_idx`` is the set of row indices that exceeded the primary's
            subword budget (live rows only). Callers route these to the fallback,
            which applies its OWN score regime. Overflow with no fallback stays
            default-zero (preserves the gender `score_B=1.0` invariant).
        `src_words`/`tgt_words` are the ORIGINAL (uncleaned) words.
        """
        from collections import defaultdict
        n = len(sources)
        out = [None] * n
        overflow_idx = set()
        groups = defaultdict(list)
        for i, s in enumerate(sources):
            groups[s if isinstance(s, str) else ""].append(i)

        for src_str, idxs in groups.items():
            src_words = self._words(src_str)
            tgt_words_per_row = {i: self._words(targets[i]) for i in idxs}
            for i in idxs:  # default: None unless successfully aligned below
                out[i] = (None, src_words, tgt_words_per_row[i])
            live = [i for i in idxs if src_words and tgt_words_per_row[i]]
            if not live:
                continue
            src_clean = [self._clean_word(w) for w in src_words]
            src_wts = self._tokenize_words(src_clean)          # compute once
            src_n = sum(len(t) for t in src_wts)
            tgt_clean_per_row = {i: [self._clean_word(w) for w in tgt_words_per_row[i]]
                                 for i in live}
            tgt_wts_per_row = {i: self._tokenize_words(tgt_clean_per_row[i]) for i in live}
            primary = [i for i in live if src_n <= self._max_subwords
                       and sum(len(t) for t in tgt_wts_per_row[i]) <= self._max_subwords]
            overflow_idx.update(set(live) - set(primary))

            if primary:
                to_encode = [src_clean] + [tgt_clean_per_row[i] for i in primary]
                wts = [src_wts] + [tgt_wts_per_row[i] for i in primary]   # reuse tokens
                vecs = self._encode(to_encode, wts=wts)
                v_src = vecs[0]
                for k, i in enumerate(primary):
                    v_tgt = vecs[1 + k]
                    if v_src.shape[0] == 0 or v_tgt.shape[0] == 0:
                        continue  # degenerate encoding → leave None (default-zero)
                    out[i] = (self._edges_from_vecs(v_src, v_tgt),
                              src_words, tgt_words_per_row[i])
                del vecs  # transient: free this group's token embeddings
        return out, overflow_idx

    # ── public API (drop-in for AlignmentScorer) ───────────────────────────────
    def _empty_result(self, return_masks):
        r = dict(bi_score=0.0, cl_score=0.0, has_bi=False, has_cl=False)
        if return_masks:
            r["bi_word_mask"] = []
            r["cl_word_mask"] = []
        return r

    def score_pair(self, source: str, target: str) -> dict:
        out = self.score_batch([source], [target], return_masks=True)
        return {
            "bi_score": float(out["bi_score"][0]),
            "cl_score": float(out["cl_score"][0]),
            "has_bi": bool(out["has_bi"][0]),
            "has_cl": bool(out["has_cl"][0]),
            "bi_word_mask": out["bi_word_mask"][0],
            "cl_word_mask": out["cl_word_mask"][0],
            "from_fallback": bool(out["from_fallback"][0]),
            "unhandled": bool(out["unhandled"][0]),
        }

    def align_batch(self, sources, targets):
        """Order-preserving word-level alignment edges per pair (batched + reused).

        Returns a list (len == len(sources)) of edge lists; each edge list is the
        aligned (source_word_index, target_word_index) pairs for that row, on the
        ORIGINAL word indexing (clean_word preserves word count). Rows with empty
        source/target (or a degenerate encoding) yield an empty edge list.

        Overflow rows are delegated to the fallback aligner's `align_batch`; with no
        fallback configured they yield an empty edge list (the gender path relies on
        this → `score_B=1.0`), and `n_overflow_unhandled` is incremented.

        Consumers that need the raw alignment, not bi/cl scores — e.g.
        `GenderConsistencyScorer` (maps src-gender[i] → tgt-gender[j]).
        """
        rows, overflow_idx = self._edges_per_row(sources, targets)
        result = [(edges if edges is not None else []) for edges, _sw, _tw in rows]
        if overflow_idx:
            self._ensure_fallback()
            if self._fallback is not None:
                oi = sorted(overflow_idx)
                fb_edges = self._fallback.align_batch([sources[i] for i in oi],
                                                      [targets[i] for i in oi])
                for k, i in enumerate(oi):
                    result[i] = fb_edges[k]
            else:
                self.n_overflow_unhandled += len(overflow_idx)
        return result

    def score_batch(self, sources, targets, return_masks: bool = True) -> dict:
        """Order-preserving batched BI/CL scoring with per-source-group reuse.

        Returns the same dict shape as `AlignmentScorer.score_batch`, plus two
        boolean arrays (always present):
          - `from_fallback`: True for rows routed to the fallback (scored under its
            own model + score regime).
          - `unhandled`: True for overflow rows that NO aligner in the chain could
            score (overflow with no fallback, or overflow exceeding even the
            fallback's budget). These carry default-zero scores → optimistic
            `bi_cl=1.0`, so downstream filters can exclude/flag them instead of
            treating them as genuine low-copy rows. `n_overflow_unhandled` is the
            running total of `unhandled` across calls.
        Overflow rows are delegated to the fallback's `score_batch` and spliced back
        by original index.
        """
        n = len(sources)
        bi = np.zeros(n)
        cl = np.zeros(n)
        from_fallback = np.zeros(n, dtype=bool)
        unhandled = np.zeros(n, dtype=bool)
        bi_masks = [[] for _ in range(n)]
        cl_masks = [[] for _ in range(n)]

        rows, overflow_idx = self._edges_per_row(sources, targets)
        for i, (edges, src_words, tgt_words) in enumerate(rows):
            # None ⇒ leave default-zero (empty/degenerate, or overflow handled below)
            if edges is None or not src_words or not tgt_words:
                continue
            aligned_src = {a for a, _ in edges}
            aligned_tgt = {b for _, b in edges}
            bi_un = [j for j in range(len(tgt_words)) if j not in aligned_tgt]
            cl_un = [j for j in range(len(src_words)) if j not in aligned_src]
            bi[i] = self._compute_score(bi_un, len(tgt_words))
            cl[i] = self._compute_score(cl_un, len(src_words))
            if return_masks:
                bi_masks[i] = [(j, tgt_words[j]) for j in bi_un]
                cl_masks[i] = [(j, src_words[j]) for j in cl_un]

        has_bi = bi >= self.bi_threshold
        has_cl = cl >= self.cl_threshold

        # Overflow rows: score under the fallback's OWN model + regime + thresholds.
        if overflow_idx:
            self._ensure_fallback()
            if self._fallback is not None:
                oi = sorted(overflow_idx)
                fb = self._fallback.score_batch([sources[i] for i in oi],
                                                [targets[i] for i in oi],
                                                return_masks=return_masks)
                fb_unhandled = fb.get("unhandled")
                for k, i in enumerate(oi):
                    bi[i] = fb["bi_score"][k]
                    cl[i] = fb["cl_score"][k]
                    has_bi[i] = fb["has_bi"][k]   # fallback's own threshold
                    has_cl[i] = fb["has_cl"][k]
                    from_fallback[i] = True
                    # propagate the fallback's give-ups (it overflowed too)
                    if fb_unhandled is not None and fb_unhandled[k]:
                        unhandled[i] = True
                    if return_masks:
                        bi_masks[i] = fb["bi_word_mask"][k]
                        cl_masks[i] = fb["cl_word_mask"][k]
            else:
                # no fallback configured → these overflow rows are unscored
                for i in overflow_idx:
                    unhandled[i] = True

        self.n_overflow_unhandled += int(unhandled.sum())

        out = dict(
            bi_score=bi,
            cl_score=cl,
            has_bi=has_bi,
            has_cl=has_cl,
            from_fallback=from_fallback,
            unhandled=unhandled,
        )
        if return_masks:
            out["bi_word_mask"] = bi_masks
            out["cl_word_mask"] = cl_masks
        return out


# Backward-compatible alias: the original class was mBERT-specific. The generalized
# BatchedAligner subsumes it; callers wanting mBERT pass model="bert", layer=9.
BatchedMBertAligner = BatchedAligner
