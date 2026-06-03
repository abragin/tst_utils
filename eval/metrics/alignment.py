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
    fallback_model : str or None
        Model to use when the primary model raises RuntimeError (typically because
        the combined source+target token length exceeds its positional embedding
        limit). Recommended: "cointegrated/rubert-tiny2" (max_position_embeddings=2048,
        F1=0.707). bert-base-multilingual-cased has the same 512 limit as BERT-base
        and is NOT a useful fallback.
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
    ):
        try:
            from simalign import SentenceAligner
        except ImportError:
            raise ImportError("simalign is required: pip install simalign")

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
            return al

        self._aligner = _make_aligner(model)
        self._fallback_aligner = _make_aligner(fallback_model) if fallback_model else None

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

    def _align_pair(self, src_words: List[str], tgt_words: List[str]):
        """Return (aligned_src_indices, aligned_tgt_indices).

        Falls back to self._fallback_aligner on RuntimeError (e.g. sequence
        length exceeds the primary model's positional embedding limit).
        """
        if not src_words or not tgt_words:
            return set(), set()
        src_clean = [self._clean_word(w) for w in src_words]
        tgt_clean = [self._clean_word(w) for w in tgt_words]
        try:
            alignments = self._aligner.get_word_aligns(src_clean, tgt_clean)
        except RuntimeError:
            if self._fallback_aligner is None:
                return set(), set()
            alignments = self._fallback_aligner.get_word_aligns(src_clean, tgt_clean)
        al = alignments[self.method]
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
        """
        src_words = source.split()
        tgt_words = target.split()

        if self.max_words_per_side is not None:
            src_words = src_words[:self.max_words_per_side]
            tgt_words = tgt_words[:self.max_words_per_side]

        if not src_words or not tgt_words:
            return dict(bi_score=0.0, cl_score=0.0, has_bi=False, has_cl=False,
                        bi_word_mask=[], cl_word_mask=[])

        aligned_src, aligned_tgt = self._align_pair(src_words, tgt_words)

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
            bi_word_mask  : list[list[tuple]] — only if return_masks=True
            cl_word_mask  : list[list[tuple]] — only if return_masks=True
        """
        bi_scores, cl_scores = [], []
        bi_masks, cl_masks = [], []

        for src, tgt in zip(sources, targets):
            r = self.score_pair(src, tgt)
            bi_scores.append(r["bi_score"])
            cl_scores.append(r["cl_score"])
            if return_masks:
                bi_masks.append(r["bi_word_mask"])
                cl_masks.append(r["cl_word_mask"])

        out = dict(
            bi_score=np.array(bi_scores),
            cl_score=np.array(cl_scores),
            has_bi=np.array(bi_scores) >= self.bi_threshold,
            has_cl=np.array(cl_scores) >= self.cl_threshold,
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
    ):
        try:
            from simalign import SentenceAligner
        except ImportError:
            raise ImportError("simalign is required: pip install simalign")

        if method != "itermax":
            raise NotImplementedError("BatchedAligner currently supports itermax only")

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

        # Lazy per-pair fallback aligner (e.g. rubert-tiny2 for overflow long texts).
        self._fallback_model = fallback_model
        self._fallback_layer = fallback_layer
        self._fallback_sa = None

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

    def _encode(self, word_lists):
        """Batched encode of pretokenized texts → list of per-word vector arrays.

        Each text's vectors are independent of batch partners (masked attention),
        so this is safe to call per source-group. Assumes inputs have already
        passed the subword-budget check in `_edges_per_row` (overflow rows are
        routed to `_fallback_edges` before reaching here); the `truncation=True`
        cap is only a defensive guard, not the long-text handling path.
        """
        import torch
        out = [None] * len(word_lists)
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

    def _n_subwords(self, clean_words):
        return sum(len(self._tok.tokenize(w)) for w in clean_words)

    def _ensure_fallback(self):
        if self._fallback_sa is not None or not self._fallback_model:
            return
        from simalign import SentenceAligner
        req = self._fallback_layer if self._fallback_layer is not None else self.layer
        sa = SentenceAligner(model=self._fallback_model, token_type="word",
                             matching_methods="i", device=self._device_str, layer=req)
        # clamp layer for small fallbacks (e.g. rubert-tiny2 has 3 layers)
        n_hidden = sa.embed_loader.emb_model.config.num_hidden_layers
        if req > n_hidden:
            sa.embed_loader.layer = n_hidden
        tok = sa.embed_loader.tokenizer
        if getattr(tok, "add_prefix_space", None) is False:
            from transformers import AutoTokenizer
            sa.embed_loader.tokenizer = AutoTokenizer.from_pretrained(
                self._fallback_model, add_prefix_space=True)
        self._fallback_sa = sa

    def _fallback_edges(self, src_words, tgt_words):
        """Aligned edges via the fallback model (cold path for overflow texts).
        Returns None when no fallback is configured or alignment fails (→ row left
        at default-zero, matching the previous behaviour)."""
        self._ensure_fallback()
        if self._fallback_sa is None:
            return None
        sc = [self._clean_word(w) for w in src_words]
        tc = [self._clean_word(w) for w in tgt_words]
        try:
            al = self._fallback_sa.get_word_aligns(sc, tc)["itermax"]
        except Exception:
            return None
        return [(int(i), int(j)) for i, j in al]

    def _edges_per_row(self, sources, targets):
        """Order-preserving per-row alignment with source-group reuse.

        Returns a list of (edges, src_words, tgt_words) per input row, where
        `edges` is the list of aligned (i, j) word-index pairs, or **None** for
        rows that must be left at default-zero (empty source/target, or a degenerate
        empty encoding). `src_words`/`tgt_words` are the ORIGINAL (uncleaned) words.
        Shared by `score_batch` (→ bi/cl) and `align_batch` (→ gender, etc.).
        """
        from collections import defaultdict
        n = len(sources)
        out = [None] * n
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
            src_n = self._n_subwords(src_clean)
            tgt_clean_per_row = {i: [self._clean_word(w) for w in tgt_words_per_row[i]]
                                 for i in live}
            primary = [i for i in live if src_n <= self._max_subwords
                       and self._n_subwords(tgt_clean_per_row[i]) <= self._max_subwords]
            overflow_set = set(live) - set(primary)

            if primary:
                to_encode = [src_clean] + [tgt_clean_per_row[i] for i in primary]
                vecs = self._encode(to_encode)
                v_src = vecs[0]
                for k, i in enumerate(primary):
                    v_tgt = vecs[1 + k]
                    if v_src.shape[0] == 0 or v_tgt.shape[0] == 0:
                        continue  # degenerate encoding → leave None (default-zero)
                    out[i] = (self._edges_from_vecs(v_src, v_tgt),
                              src_words, tgt_words_per_row[i])
                del vecs  # transient: free this group's token embeddings

            for i in overflow_set:
                edges = self._fallback_edges(src_words, tgt_words_per_row[i])
                out[i] = (edges, src_words, tgt_words_per_row[i])
        return out

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
        }

    def align_batch(self, sources, targets):
        """Order-preserving word-level alignment edges per pair (batched + reused).

        Returns a list (len == len(sources)) of edge lists; each edge list is the
        aligned (source_word_index, target_word_index) pairs for that row, on the
        ORIGINAL word indexing (clean_word preserves word count). Rows with empty
        source/target (or a degenerate encoding) yield an empty edge list.

        Consumers that need the raw alignment, not bi/cl scores — e.g.
        `GenderConsistencyScorer` (maps src-gender[i] → tgt-gender[j]).
        """
        rows = self._edges_per_row(sources, targets)
        return [(edges if edges is not None else []) for edges, _sw, _tw in rows]

    def score_batch(self, sources, targets, return_masks: bool = True) -> dict:
        """Order-preserving batched BI/CL scoring with per-source-group reuse.

        Returns the same dict shape as `AlignmentScorer.score_batch`.
        """
        n = len(sources)
        bi = np.zeros(n)
        cl = np.zeros(n)
        bi_masks = [[] for _ in range(n)]
        cl_masks = [[] for _ in range(n)]

        for i, (edges, src_words, tgt_words) in enumerate(self._edges_per_row(sources, targets)):
            # None ⇒ leave default-zero (empty source/target or degenerate encoding)
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

        out = dict(
            bi_score=bi,
            cl_score=cl,
            has_bi=bi >= self.bi_threshold,
            has_cl=cl >= self.cl_threshold,
        )
        if return_masks:
            out["bi_word_mask"] = bi_masks
            out["cl_word_mask"] = cl_masks
        return out


# Backward-compatible alias: the original class was mBERT-specific. The generalized
# BatchedAligner subsumes it; callers wanting mBERT pass model="bert", layer=9.
BatchedMBertAligner = BatchedAligner
