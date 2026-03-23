"""
BI/CL span removal using a soft ensemble of SimAlign density and a fine-tuned token classifier.

Usage:
    from tst_utils.eval.bi_cl_cleaner import BiClEnsemble, remove_bi_span_greedy

    ensemble = BiClEnsemble(model_path="path/to/bi_cl_model/best", device="cuda")

    # Full pipeline: detect + clean
    result = ensemble.score_and_clean_pair(source, target)
    # result keys: bi_word_scores, cl_word_scores, has_bi, has_cl, cleaned_target

    # Or use the parts separately:
    scores = ensemble.score_pair(source, target)
    cleaned = remove_bi_span_greedy(target, scores["bi_word_scores"])

Ensemble:
    bi_word_scores[i] = α * density_simalign[i] + (1−α) * prob_finetuned[i]
    Best config (tuned on dev set): α=0.80, threshold=0.45  (BI F1=0.779, CL F1=0.766)

Removal strategy (v2 — greedy sub-sentence):
    Every removal span is within a single sentence and anchored at one sentence
    boundary (start or end).  The other end may fall at any sub-sentence split
    point from the chosen boundary set (sentence, semicolon, em-dash, comma, or
    singledash " - ").  This prevents removing mid-sentence fragments where both
    ends are punctuation marks (e.g. "и Амура,").

    Winner config from real-data quality sweep (notebook 07):
        boundaries='sentence+semicolon+dash+comma', sent_threshold=0.40
        combined score 0.727 (0.5·norm(Δmeaning) + 0.5·norm(Δbi))
        Δmeaning=+0.0154, Δbi=+0.4995, 78% of has_bi changed

    Production recommendation: use select_best_removal() or score_and_clean_pair(use_hybrid=True),
    which runs both greedy/0.40 and longest/0.45 configs and returns the one with lowest
    bi_score + cl_peak on the cleaned text.

    Post-processing:
        - End-anchored partial removal: strip trailing split punctuation (, ; — - spaces)
          left at the join point.
        - Start-anchored partial removal: capitalise the first letter of the remaining
          sentence fragment.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)

# Best ensemble hyper-params from dev-set sweep (notebook 06)
DEFAULT_ALPHA = 0.80
DEFAULT_THRESHOLD = 0.45
DEFAULT_SA_WINDOW = 15
DEFAULT_GATE_THRESHOLD = 0.75  # AlignmentScorer rolling_max default (nb04)
DEFAULT_GATE_WINDOW = 9        # rolling_max window matching nb04 best config
DEFAULT_MAX_WORDS = 512
DEFAULT_MAX_LEN = 1024

# Best span-removal config from real-data quality sweep (notebook 07)
DEFAULT_SENT_THRESHOLD = 0.40
DEFAULT_BOUNDARIES = 'sentence+semicolon+dash+comma'

# Characters to strip at split-point boundaries after end-anchored partial removal.
# Covers: space, comma, semicolon, em-dash (U+2014), ASCII hyphen.
_TRAIL = frozenset(' ,-;\u2014')


# ---------------------------------------------------------------------------
# Greedy span removal helpers
# ---------------------------------------------------------------------------

def _sentence_sub_segs(sent_text, sent_start, boundaries):
    """Return (abs_start, abs_end) char spans for sub-segments of one sentence."""
    use_semi       = 'semicolon'   in boundaries
    use_dash       = 'dash'        in boundaries and 'singledash' not in boundaries
    use_comma      = 'comma'       in boundaries
    use_singledash = 'singledash'  in boundaries
    if boundaries == 'sentence+all':
        use_semi = use_dash = use_comma = use_singledash = True

    split_pts = [0]
    if use_semi:        [split_pts.append(m.end()) for m in re.finditer(r';',      sent_text)]
    if use_dash:        [split_pts.append(m.end()) for m in re.finditer(r' -- |—', sent_text)]
    if use_comma:       [split_pts.append(m.end()) for m in re.finditer(r',',      sent_text)]
    if use_singledash:  [split_pts.append(m.end()) for m in re.finditer(r' - ',    sent_text)]
    split_pts.append(len(sent_text))
    split_pts = sorted(set(split_pts))
    return [
        (sent_start + split_pts[k], sent_start + split_pts[k + 1])
        for k in range(len(split_pts) - 1)
        if split_pts[k + 1] > split_pts[k]
    ]


def _build_candidates(target, scores, cw, sentences, boundaries):
    """Sorted-descending list of (mean_score, char_start, char_end) candidates.

    Spans are within one sentence and anchored at a sentence boundary on one end.
    Uses prefix sums for O(n_segs) per sentence.
    """
    n = len(scores)
    candidates = []
    for sent in sentences:
        segs = _sentence_sub_segs(sent.text, sent.start, boundaries)
        if not segs:
            continue
        nseg = len(segs)
        seg_s_sum = np.zeros(nseg, dtype=float)
        seg_w_cnt = np.zeros(nseg, dtype=int)
        for k, (seg_s, seg_e) in enumerate(segs):
            widxs = sorted({cw[i] for i in range(seg_s, min(seg_e, len(cw) - 1))
                            if 0 <= cw[i] < n})
            if widxs:
                seg_s_sum[k] = scores[widxs].sum()
                seg_w_cnt[k] = len(widxs)
        ps = np.zeros(nseg + 1, dtype=float); ps[1:] = np.cumsum(seg_s_sum)
        pc = np.zeros(nseg + 1, dtype=int);   pc[1:] = np.cumsum(seg_w_cnt)
        cs0, ce0 = segs[0][0], segs[-1][1]
        for j in range(nseg):                          # start-anchored
            cnt = pc[j + 1]
            if cnt:
                candidates.append((ps[j + 1] / cnt, cs0, segs[j][1]))
        for i in range(nseg):                          # end-anchored
            cnt = pc[nseg] - pc[i]
            if cnt:
                candidates.append(((ps[nseg] - ps[i]) / cnt, segs[i][0], ce0))
    candidates.sort(key=lambda x: -x[0])
    return candidates


def _greedy_select(candidates, sent_thr):
    """Return (char_start, char_end) spans from a sorted candidate list."""
    removed = []
    for score, cs, ce in candidates:
        if score <= sent_thr:
            break  # sorted descending — no higher scores remain
        if not any(max(cs, rs) < min(ce, re) for rs, re in removed):
            removed.append((cs, ce))
    return removed


def _longest_select(scores, cw, sentences, boundaries, sent_thr):
    """Per-sentence: pick the longest span with mean BI score > sent_thr.

    Tie-break: longest span (by char count) then highest mean score.
    Unlike greedy selection, each sentence contributes at most one span.
    """
    n = len(scores)
    removed = []
    for sent in sentences:
        segs = _sentence_sub_segs(sent.text, sent.start, boundaries)
        if not segs:
            continue
        nseg = len(segs)
        seg_s_sum = np.zeros(nseg, dtype=float)
        seg_w_cnt = np.zeros(nseg, dtype=int)
        for k, (seg_s, seg_e) in enumerate(segs):
            widxs = sorted({cw[i] for i in range(seg_s, min(seg_e, len(cw) - 1))
                            if 0 <= cw[i] < n})
            if widxs:
                seg_s_sum[k] = scores[widxs].sum()
                seg_w_cnt[k] = len(widxs)
        ps = np.zeros(nseg + 1, dtype=float); ps[1:] = np.cumsum(seg_s_sum)
        pc = np.zeros(nseg + 1, dtype=int);   pc[1:] = np.cumsum(seg_w_cnt)
        cs0, ce0 = segs[0][0], segs[-1][1]
        sent_cands = []
        for j in range(nseg):          # start-anchored
            cnt = pc[j + 1]
            if cnt:
                sent_cands.append((ps[j + 1] / cnt, cs0, segs[j][1]))
        for i in range(nseg):          # end-anchored
            cnt = pc[nseg] - pc[i]
            if cnt:
                sent_cands.append(((ps[nseg] - ps[i]) / cnt, segs[i][0], ce0))
        valid = [(sc, cs, ce) for sc, cs, ce in sent_cands if sc > sent_thr]
        if not valid:
            continue
        best = max(valid, key=lambda x: (x[2] - x[1], x[0]))  # longest, then highest score
        removed.append((best[1], best[2]))
    return removed


def _find_removed_ranges(target, bi_word_scores, sent_thr, boundaries,
                         strategy='greedy_density'):
    """Full pipeline: char→word map + sentenize + selection.

    strategy : 'greedy_density'    — sort all candidates by mean score, pick greedily.
               'longest_above_thr' — per sentence, pick the longest span above threshold.
    """
    try:
        from razdel import sentenize
    except ImportError:
        raise ImportError("razdel is required: pip install razdel")

    words = target.split()
    if not words or len(bi_word_scores) == 0:
        return []
    n      = min(len(bi_word_scores), len(words))
    scores = np.array(bi_word_scores[:n], dtype=float)
    cw     = np.full(len(target) + 1, -1, dtype=int)
    pos    = 0
    for widx, word in enumerate(words[:n]):
        while pos < len(target) and target[pos].isspace():
            pos += 1
        end = min(pos + len(word), len(target))
        cw[pos:end] = widx
        pos = end
    sentences = list(sentenize(target))
    if strategy == 'greedy_density':
        candidates = _build_candidates(target, scores, cw, sentences, boundaries)
        return _greedy_select(candidates, sent_thr)
    elif strategy == 'longest_above_thr':
        return _longest_select(scores, cw, sentences, boundaries, sent_thr)
    else:
        raise ValueError(f"Unknown removal strategy: {strategy!r}")


def remove_bi_span_greedy(
    target: str,
    bi_word_scores: np.ndarray,
    sent_thr: float = DEFAULT_SENT_THRESHOLD,
    boundaries: str = DEFAULT_BOUNDARIES,
    strategy: str = 'greedy_density',
) -> str:
    """Remove BI spans from target text using greedy sub-sentence selection.

    Every removal span is within a single sentence and anchored at one sentence
    boundary.  The other end may fall at any split point in the chosen boundary
    set (``sentence``, ``sentence+dash``, ``sentence+comma``, etc.).

    Post-processing for partial-sentence removals:
    - End-anchored partial (sentence tail removed): strips trailing split
      punctuation (, ; — - and spaces) left at the join point.
    - Start-anchored partial (sentence head removed): capitalises the first
      letter of the remaining sentence fragment.

    Parameters
    ----------
    target : str
    bi_word_scores : array-like
        Per-word BI scores, length == len(target.split()).
    sent_thr : float
        Score threshold above which a span is removed.  Default: 0.40.
    boundaries : str
        Boundary set.  One of: ``sentence``, ``sentence+dash``,
        ``sentence+semicolon``, ``sentence+comma``, ``sentence+singledash``,
        ``sentence+semicolon+dash``, ``sentence+semicolon+dash+comma``,
        ``sentence+comma+singledash``, ``sentence+all``.
        Default: ``sentence+semicolon+dash+comma``.

    Returns
    -------
    str
        Cleaned text, or the original text unchanged if nothing exceeds the threshold.
    """
    try:
        from razdel import sentenize
    except ImportError:
        raise ImportError("razdel is required: pip install razdel")

    removed = _find_removed_ranges(target, bi_word_scores, sent_thr, boundaries, strategy)
    if not removed:
        return target

    sentences = list(sentenize(target))

    sent_starts = {s.start for s in sentences}
    sent_ends   = {s.stop  for s in sentences}

    chars = list(target)       # mutable for capitalisation
    keep  = [True] * len(target)
    for s, e in removed:
        for i in range(s, e):
            keep[i] = False

    for span_s, span_e in removed:
        is_sent_start = span_s in sent_starts
        is_sent_end   = span_e in sent_ends
        if is_sent_start and is_sent_end:
            continue  # full-sentence removal — no fix needed

        if is_sent_end and not is_sent_start:
            # End-anchored partial: strip split-char residue before the gap
            pos = span_s - 1
            while pos >= 0 and not keep[pos]:
                pos -= 1
            while pos >= 0 and keep[pos] and chars[pos] in _TRAIL:
                keep[pos] = False
                pos -= 1

        if is_sent_start and not is_sent_end:
            # Start-anchored partial: capitalise first kept letter after the gap
            pos = span_e
            while pos < len(target) and (not keep[pos] or chars[pos] == ' '):
                pos += 1
            if pos < len(target) and keep[pos] and chars[pos].islower():
                chars[pos] = chars[pos].upper()

    return ''.join(c for c, k in zip(chars, keep) if k).strip()


# Default configs for the hybrid selector (tuned on real-data comparison, notebook 07).
_HYBRID_CONFIGS = [
    (0.40, 'sentence+semicolon+dash+comma', 'greedy_density'),
    (0.45, 'sentence+semicolon+dash+comma', 'longest_above_thr'),
]


def select_best_removal(
    source: str,
    target: str,
    bi_word_scores: np.ndarray,
    ensemble: "BiClEnsemble",
    configs=None,
) -> str:
    """Run multiple removal configs; return the one with lowest bi_score + cl_peak.

    Each cleaned candidate is rescored via ``ensemble.score_pair(source, cleaned)``.
    The combined score ``bi_score + max(cl_word_scores)`` measures remaining injection
    plus content loss induced by the removal — lower is better.

    Parameters
    ----------
    source, target : str
    bi_word_scores : array-like
        Per-word BI scores for target (from ``ensemble.score_pair``).
    ensemble : BiClEnsemble
        Used to rescore each cleaned candidate.
    configs : list of (sent_thr, boundaries, strategy) tuples, optional
        Removal configs to try.  Defaults to ``_HYBRID_CONFIGS``
        (greedy/0.40 and longest/0.45, both with sentence+semicolon+dash+comma).

    Returns
    -------
    str
        Cleaned text with lowest bi_score + cl_peak, or ``target`` unchanged if
        no config produces a removal.
    """
    if configs is None:
        configs = _HYBRID_CONFIGS

    seen: dict = {}   # cleaned_text → combined score
    order = []
    for sent_thr, boundaries, strategy in configs:
        cleaned = remove_bi_span_greedy(target, bi_word_scores, sent_thr, boundaries, strategy)
        if not cleaned or cleaned == target or cleaned in seen:
            continue
        sc = ensemble.score_pair(source, cleaned)
        cl_peak = float(np.max(sc["cl_word_scores"])) if len(sc["cl_word_scores"]) > 0 else 0.0
        seen[cleaned] = sc["bi_score"] + cl_peak
        order.append(cleaned)

    if not order:
        return target

    return min(order, key=lambda c: seen[c])


def remove_bi_span(
    target: str,
    bi_word_scores: np.ndarray,
    sent_threshold: float = 0.5,
) -> str:
    """Remove BI-injected sentences from target text (sentence-level, v1).

    .. deprecated::
        Use :func:`remove_bi_span_greedy` or :func:`select_best_removal` instead.
        This function uses coarse sentence-level removal only.  The validated
        production path is ``score_and_clean_pair(use_hybrid=True)``.

    A sentence is removed if the mean BI score of its words exceeds ``sent_threshold``.
    """
    warnings.warn(
        "remove_bi_span() is deprecated and will be removed in a future version. "
        "Use remove_bi_span_greedy() or score_and_clean_pair(use_hybrid=True) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from razdel import sentenize
    except ImportError:
        raise ImportError("razdel is required: pip install razdel")

    words = target.split()
    if len(words) == 0 or len(bi_word_scores) == 0:
        return target

    scores = np.array(bi_word_scores)
    if len(scores) != len(words):
        LOG.warning(
            "bi_word_scores length %d != target word count %d; truncating to min",
            len(scores), len(words),
        )
        n = min(len(scores), len(words))
        scores = scores[:n]
        words = words[:n]

    char_to_word = np.full(len(target) + 1, -1, dtype=int)
    pos = 0
    for widx, word in enumerate(words):
        while pos < len(target) and target[pos] == ' ':
            pos += 1
        start = pos
        end = start + len(word)
        if end > len(target):
            end = len(target)
        char_to_word[start:end] = widx
        pos = end

    kept = []
    for sent in sentenize(target):
        s, e = sent.start, sent.stop
        word_indices = sorted(set(
            char_to_word[i] for i in range(s, min(e, len(char_to_word) - 1))
            if char_to_word[i] >= 0
        ))
        if not word_indices:
            kept.append(sent.text)
            continue
        mean_score = float(scores[word_indices].mean())
        if mean_score <= sent_threshold:
            kept.append(sent.text)

    return " ".join(kept)


class BiClEnsemble:
    """Soft ensemble of SimAlign density and fine-tuned rubert-tiny2 token classifier.

    Produces per-word BI/CL score arrays that can be passed directly to
    remove_bi_span_greedy().

    Parameters
    ----------
    model_path : str or Path
        Path to the fine-tuned token classifier checkpoint directory (e.g. data/bi_cl_model/best).
    alpha : float
        Weight of the SimAlign density component (1−alpha = fine-tuned weight). Default: 0.80.
    threshold : float
        Score threshold for has_bi / has_cl flags. Default: 0.45.
    sa_window : int
        Rolling density window width for the ensemble SimAlign component. Default: 15.
    gate_threshold : float
        AlignmentScorer bi_score threshold (rolling_max, w=gate_window) above which
        score_and_clean_pair() will run the full ensemble and removal. Matches the
        nb04 AlignmentScorer default. Default: 0.75.
    gate_window : int
        Rolling window for the gate bi_score (nb04 default: 9). Default: 9.
    max_words : int
        Truncate source/target to this many words before alignment. Default: 512.
    max_len : int
        Max token length for the fine-tuned model. Default: 1024.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        alpha: float = DEFAULT_ALPHA,
        threshold: float = DEFAULT_THRESHOLD,
        sa_window: int = DEFAULT_SA_WINDOW,
        gate_threshold: float = DEFAULT_GATE_THRESHOLD,
        gate_window: int = DEFAULT_GATE_WINDOW,
        max_words: int = DEFAULT_MAX_WORDS,
        max_len: int = DEFAULT_MAX_LEN,
        device: str = "cuda",
    ):
        try:
            from simalign import SentenceAligner
        except ImportError:
            raise ImportError("simalign is required: pip install simalign")
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
        except ImportError:
            raise ImportError("transformers is required: pip install transformers")

        from tst_utils.eval.metrics.alignment import AlignmentScorer

        self.alpha = alpha
        self.threshold = threshold
        self.sa_window = sa_window
        self.gate_threshold = gate_threshold
        self.gate_window = gate_window
        self.max_words = max_words
        self.max_len = max_len
        self.device = device
        self._AlignmentScorer = AlignmentScorer

        # SimAlign — rubert-tiny2, layer=3 (model has only 4 layers; default 8 crashes)
        self._sa_aligner = SentenceAligner(
            model="cointegrated/rubert-tiny2",
            token_type="word",
            matching_methods="i",   # itermax
            device=device,
            layer=3,
        )

        # Fine-tuned token classifier
        model_path = Path(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self._model.eval().to(device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_word(w: str) -> str:
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', w, flags=re.UNICODE)
        return cleaned if cleaned else w

    def _simalign_indices(self, src_words, tgt_words):
        """Return (aligned_src_indices, aligned_tgt_indices) sets."""
        if not src_words or not tgt_words:
            return set(), set()
        alignments = self._sa_aligner.get_word_aligns(
            [self._clean_word(w) for w in src_words],
            [self._clean_word(w) for w in tgt_words],
        )
        al = alignments["itermax"]
        return {i for i, j in al}, {j for i, j in al}

    def _ft_word_probs(self, src_words, tgt_words):
        """Return (bi_probs, cl_probs) as numpy float arrays, one value per word."""
        tokenizer = self._tokenizer
        sw = src_words[:self.max_words]
        tw = tgt_words[:self.max_words]

        se = tokenizer(sw, is_split_into_words=True, add_special_tokens=False)
        te = tokenizer(tw, is_split_into_words=True, add_special_tokens=False)

        ids = (
            [tokenizer.cls_token_id]
            + se["input_ids"]
            + [tokenizer.sep_token_id]
            + te["input_ids"]
            + [tokenizer.sep_token_id]
        )
        tts = (
            [0] + [0] * len(se["input_ids"]) + [0]
            + [1] * len(te["input_ids"]) + [1]
        )
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            tts = tts[:self.max_len]

        iid = torch.tensor([ids], dtype=torch.long).to(self.device)
        tti = torch.tensor([tts], dtype=torch.long).to(self.device)
        amk = torch.ones_like(iid)

        with torch.no_grad():
            logits = self._model(
                input_ids=iid, token_type_ids=tti, attention_mask=amk
            ).logits.cpu()
        probs = F.softmax(logits, dim=-1)[0, :, 1].numpy()  # shape: (seq_len,)

        tgt_start = 1 + len(se["input_ids"]) + 1

        # CL: source-side (token_type=0, after CLS)
        cl_p = np.zeros(len(sw))
        seen: set = set()
        for ti, wid in enumerate(se.word_ids()):
            if wid is not None and (1 + ti) < len(probs) and wid not in seen:
                cl_p[wid] = probs[1 + ti]
                seen.add(wid)

        # BI: target-side (token_type=1)
        bi_p = np.zeros(len(tw))
        seen = set()
        for ti, wid in enumerate(te.word_ids()):
            if wid is not None and (tgt_start + ti) < len(probs) and wid not in seen:
                bi_p[wid] = probs[tgt_start + ti]
                seen.add(wid)

        return bi_p, cl_p

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_pair(self, source: str, target: str) -> dict:
        """Score a (source, target) pair.

        Returns
        -------
        dict with:
            bi_score       : float       — doc-level BI score matching AlignmentScorer
                                           rolling_max (w=gate_window). Same metric as nb04.
                                           Use this to decide whether to attempt removal.
            bi_word_scores : np.ndarray  — per-word ensemble BI scores, length = n_target_words
            cl_word_scores : np.ndarray  — per-word ensemble CL scores, length = n_source_words
            has_bi         : bool        — bi_score >= gate_threshold
            has_cl         : bool        — max(cl_word_scores) >= threshold
        """
        src_words = source.split()[:self.max_words]
        tgt_words = target.split()[:self.max_words]
        nt, ns = len(tgt_words), len(src_words)

        if nt == 0 or ns == 0:
            return dict(
                bi_score=0.0,
                bi_word_scores=np.zeros(nt),
                cl_word_scores=np.zeros(ns),
                has_bi=False,
                has_cl=False,
            )

        # SimAlign alignment indices (one forward pass, reused for both gate and ensemble)
        aligned_src, aligned_tgt = self._simalign_indices(src_words, tgt_words)
        bi_unaligned = [j for j in range(nt) if j not in aligned_tgt]
        cl_unaligned = [i for i in range(ns) if i not in aligned_src]

        # Gate score: rolling_max(w=gate_window) — matches nb04 AlignmentScorer
        bi_score = self._AlignmentScorer._rolling_density_score(bi_unaligned, nt, self.gate_window)

        # Ensemble SA density arrays (w=sa_window, used for per-word removal)
        bi_sa = self._AlignmentScorer.rolling_density_map(bi_unaligned, nt, self.sa_window)
        cl_sa = self._AlignmentScorer.rolling_density_map(cl_unaligned, ns, self.sa_window)

        # Fine-tuned probabilities
        bi_ft, cl_ft = self._ft_word_probs(src_words, tgt_words)

        def _pad(arr, n):
            return arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)))

        bi_word_scores = self.alpha * _pad(bi_sa, nt) + (1 - self.alpha) * _pad(bi_ft, nt)
        cl_word_scores = self.alpha * _pad(cl_sa, ns) + (1 - self.alpha) * _pad(cl_ft, ns)

        return dict(
            bi_score=bi_score,
            bi_word_scores=bi_word_scores,
            cl_word_scores=cl_word_scores,
            has_bi=bool(bi_score >= self.gate_threshold),
            has_cl=bool(float(np.max(cl_word_scores)) >= self.threshold),
        )

    def score_and_clean_pair(
        self,
        source: str,
        target: str,
        sent_threshold: float = DEFAULT_SENT_THRESHOLD,
        boundaries: str = DEFAULT_BOUNDARIES,
        strategy: str = 'greedy_density',
        use_hybrid: bool = False,
    ) -> dict:
        """Score pair and return cleaned target alongside scores.

        Parameters
        ----------
        source, target : str
        sent_threshold : float
            BI score threshold for span removal.  Default: 0.40 (validated winner).
        boundaries : str
            Sub-sentence boundary set for span selection.
            Default: ``sentence+semicolon+dash+comma`` (validated winner).
        strategy : str
            Removal strategy: ``'greedy_density'`` or ``'longest_above_thr'``.
            Ignored when ``use_hybrid=True``.
        use_hybrid : bool
            If True, run both greedy/0.40 and longest/0.45 configs and return the
            one with lowest ``bi_score + cl_peak`` (see ``select_best_removal``).

        Returns
        -------
        dict — score_pair() result plus:
            cleaned_target : str — target with high-BI spans removed, or original
                             target unchanged when has_bi is False
                             (i.e. bi_score < gate_threshold).
        """
        result = self.score_pair(source, target)
        if result["has_bi"]:
            if use_hybrid:
                result["cleaned_target"] = select_best_removal(
                    source, target, result["bi_word_scores"], self
                )
            else:
                result["cleaned_target"] = remove_bi_span_greedy(
                    target, result["bi_word_scores"], sent_threshold, boundaries, strategy
                )
        else:
            result["cleaned_target"] = target
        return result
