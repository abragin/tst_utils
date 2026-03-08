"""
BI/CL detection via word alignment (SimAlign).

Boilerplate Injection (BI): target words with no alignment to source.
Content Loss (CL): source words with no alignment to target.

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
        self._aligner = SentenceAligner(
            model=model,
            token_type="word",
            matching_methods=self._method_char(method),
            device=device,
        )

    def _compute_score(self, unaligned_indices: list, total: int) -> float:
        """Dispatch to the configured scoring function."""
        if self.score_fn == "rolling_max":
            return self._rolling_density_score(unaligned_indices, total, self.window)
        return self._span_score(unaligned_indices, total)

    @staticmethod
    def _rolling_density_score(unaligned_indices: list, total: int, window: int = 9) -> float:
        """Peak local unalignment density (uniform kernel, edge-normalised).

        density[i] = fraction of words in the window centred on i that are unaligned,
        normalised by actual in-bounds window size so edge positions are not penalised.
        bi_score = max(density) over all positions.
        """
        if total == 0 or not unaligned_indices:
            return 0.0
        unaligned = np.zeros(total)
        for i in unaligned_indices:
            if 0 <= i < total:
                unaligned[i] = 1.0
        k = np.ones(window) / window
        norm    = np.convolve(np.ones(total), k, mode="same")
        density = np.convolve(unaligned, k, mode="same") / norm
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
        """Return (aligned_src_indices, aligned_tgt_indices)."""
        if not src_words or not tgt_words:
            return set(), set()
        alignments = self._aligner.get_word_aligns(
            [self._clean_word(w) for w in src_words],
            [self._clean_word(w) for w in tgt_words],
        )
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
