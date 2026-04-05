"""
Gender Consistency Metric (Task 1.6).

Detects narrator gender switches between source and paraphrase texts.
A gender switch occurs when the PPH model flips the narrator's grammatical
gender (e.g. 'она написала' → 'он написал').

Score convention: 1.0 = no switch detected, 0.0 = definite switch.
A pair is flagged as a switch when score < threshold (recommended: 0.80).

Method
------
Combined score = mean(score_A, score_B):

  Approach A (distribution, PRON-activated):
    Collect gender-tagged tokens — PRON+Sing, VERB+Past+Sing, ADJ+Short — from
    source and target. Activation gate: fire only when source contains at least
    one PRON+Number=Sing (best available proxy for singular narrator, since
    rupostagger has no Animacy tags). Score = 1 - |p_fem_src - p_fem_tgt|;
    returns 1.0 if no activation.

  Approach B (hard alignment, SimAlign bert-multilingual):
    Align source/target word pairs with itermax. For each aligned pair where
    BOTH words carry a Gender= tag (Masc or Fem), count agreements vs
    mismatches. Score = n_agree / n_gendered_pairs (1.0 if none).

Validation (gender_val_clean.csv, n=371, 34 LLM-labeled positives):
  A alone:       AUC=0.821, F1=0.471
  B alone:       AUC=0.811, F1=0.378
  mean(A, B):    AUC=0.863, F1=0.418   ← used here
  (Note: earlier notebook computation showed 0.894, which was an artifact of
   computing score_A on styled_text while score_B used styled_cleaned.
   0.863 is the self-consistent figure with both scores on the same target.)

Dependencies: rupostagger, razdel, simalign (bert-base-multilingual-cased).
"""

import logging
import re
from typing import List

import numpy as np

LOG = logging.getLogger(__name__)


class GenderConsistencyScorer:
    """
    Gender-switch detector combining distribution (A) and alignment (B) signals.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'. Passed to the AlignmentScorer (bert-multilingual).

    Examples
    --------
    >>> scorer = GenderConsistencyScorer()
    >>> r = scorer.score_pair("Она написала письмо.", "Он написал письмо.")
    >>> r['gender_score']      # 0.0 — clear switch
    0.0
    >>> r = scorer.score_pair("Она написала письмо.", "Она написала письмо.")
    >>> r['gender_score']      # 1.0 — no switch
    1.0
    """

    SWITCH_THRESHOLD = 0.80

    def __init__(self, device: str = "cuda"):
        self._tagger = self._load_tagger()
        self._aligner = self._load_aligner(device)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_tagger():
        try:
            from rupostagger import RuPosTagger
        except ImportError:
            raise ImportError(
                "rupostagger is required. Install from "
                "https://github.com/Koziev/rupostagger"
            )
        tagger = RuPosTagger()
        tagger.load()
        return tagger

    @staticmethod
    def _load_aligner(device: str):
        from tst_utils.eval.metrics.alignment import AlignmentScorer
        return AlignmentScorer(
            model="bert",
            method="itermax",
            score_fn="span",
            min_span=1,
            device=device,
        )

    def _tag_text(self, text: str):
        """Return list of (word_str, tag_or_None) using razdel tokenisation."""
        import razdel
        tokens = [t.text for t in razdel.tokenize(text)]
        tagged = list(self._tagger.tag(tokens))
        return [(w, t[1] if t[1] else None) for w, t in zip(tokens, tagged)]

    def _extract_gender_tokens(self, text: str):
        """
        Return (pron_present, score_tokens).

        pron_present : True if at least one PRON+Number=Sing is in the text.
        score_tokens : list of (word, gender_str) for gender-bearing tokens:
            PRON+Sing with Gender=, VERB+Past+Sing, ADJ+Short.
        """
        pron_present = False
        score_tokens = []
        for word, tag in self._tag_text(text):
            if not tag:
                continue
            parts = tag.split("|")
            pos = parts[0]
            attrs = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            gender = attrs.get("Gender")

            if pos == "PRON" and attrs.get("Number") == "Sing":
                pron_present = True
                if gender:
                    score_tokens.append((word, gender))
            elif pos == "VERB" and attrs.get("Tense") == "Past" and attrs.get("Number") == "Sing":
                if gender:
                    score_tokens.append((word, gender))
            elif pos == "ADJ" and attrs.get("Variant") == "Short":
                if gender:
                    score_tokens.append((word, gender))
        return pron_present, score_tokens

    @staticmethod
    def _distribution_score(src_tokens, tgt_tokens) -> float:
        """1 - |p_fem_src - p_fem_tgt|. Returns 1.0 if no gender tokens."""
        def p_fem(tokens):
            total = sum(1 for _, g in tokens if g in ("Masc", "Fem"))
            if total == 0:
                return None
            return sum(1 for _, g in tokens if g == "Fem") / total

        p_src = p_fem(src_tokens)
        p_tgt = p_fem(tgt_tokens)
        if p_src is None and p_tgt is None:
            return 1.0
        p_src = p_src if p_src is not None else 0.5
        p_tgt = p_tgt if p_tgt is not None else 0.5
        return 1.0 - abs(p_src - p_tgt)

    def _build_gender_map(self, text: str) -> dict:
        """
        Map whitespace-word index (text.split()) → 'Masc' or 'Fem'.

        Uses razdel character offsets to bridge razdel tokens (which split
        punctuation) and whitespace words (which SimAlign and text.split() use).
        """
        import razdel
        words = text.split()
        if not words:
            return {}
        # Build character spans for each whitespace word
        word_spans = []
        pos = 0
        for word in words:
            start = text.index(word, pos)
            word_spans.append((start, start + len(word)))
            pos = start + len(word)

        rtokens = list(razdel.tokenize(text))
        tagged = self._tag_text(text)  # (word_str, tag_or_None), razdel-indexed
        gmap = {}
        for rtok, (_, tag) in zip(rtokens, tagged):
            if not tag:
                continue
            attrs = dict(p.split("=", 1) for p in tag.split("|")[1:] if "=" in p)
            g = attrs.get("Gender")
            if g not in ("Masc", "Fem"):
                continue
            for i, (ws, we) in enumerate(word_spans):
                if rtok.start >= ws and rtok.stop <= we:
                    gmap[i] = g
                    break
        return gmap

    def _score_A(self, source: str, target: str) -> float:
        """Approach A: distribution-based, PRON-activated."""
        src_pron, src_tokens = self._extract_gender_tokens(source)
        _, tgt_tokens = self._extract_gender_tokens(target)
        if not src_pron:
            return 1.0
        return self._distribution_score(src_tokens, tgt_tokens)

    def _score_B(self, source: str, target: str) -> float:
        """Approach B: hard alignment (SimAlign bert-multilingual, itermax)."""
        src_words = source.split()
        tgt_words = target.split()
        if not src_words or not tgt_words:
            return 1.0

        src_gmap = self._build_gender_map(source)
        tgt_gmap = self._build_gender_map(target)
        if not src_gmap and not tgt_gmap:
            return 1.0

        try:
            al = self._aligner._aligner.get_word_aligns(
                [self._aligner._clean_word(w) for w in src_words],
                [self._aligner._clean_word(w) for w in tgt_words],
            )[self._aligner.method]
        except (IndexError, RuntimeError):
            # Very long texts (>512 subword tokens) cause simalign IndexError.
            # score_B falls back to 1.0 (no switch detected); only score_A contributes.
            LOG.warning(
                "score_B fallback: alignment failed for text of length %d/%d words "
                "(likely >512 subword tokens). Returning score_B=1.0.",
                len(src_words), len(tgt_words),
            )
            return 1.0

        n_gendered = n_agree = 0
        for i, j in al:
            sg = src_gmap.get(i)
            tg = tgt_gmap.get(j)
            if sg and tg:
                n_gendered += 1
                if sg == tg:
                    n_agree += 1
        return 1.0 if n_gendered == 0 else n_agree / n_gendered

    # ── public interface ──────────────────────────────────────────────────────

    def score_pair(self, source: str, target: str) -> dict:
        """
        Score a single (source, target) pair.

        Returns
        -------
        dict with:
            gender_score : float — combined score, 1.0=no switch, 0.0=switch
            score_A      : float — Approach A (distribution) score
            score_B      : float — Approach B (alignment) score
            activated    : bool  — True if source contains PRON+Sing (A fired)
            has_switch   : bool  — gender_score < SWITCH_THRESHOLD (0.80)
        """
        if not source.split() or not target.split():
            return dict(gender_score=1.0, score_A=1.0, score_B=1.0,
                        activated=False, has_switch=False)
        score_a = self._score_A(source, target)
        score_b = self._score_B(source, target)
        combined = (score_a + score_b) / 2.0
        src_pron, _ = self._extract_gender_tokens(source)
        return dict(
            gender_score=combined,
            score_A=score_a,
            score_B=score_b,
            activated=src_pron,
            has_switch=combined < self.SWITCH_THRESHOLD,
        )

    def score_batch(
        self,
        sources: List[str],
        targets: List[str],
    ) -> dict:
        """
        Score a batch of (source, target) pairs.

        Returns
        -------
        dict with numpy arrays:
            gender_score : np.ndarray[float]
            score_A      : np.ndarray[float]
            score_B      : np.ndarray[float]
            activated    : np.ndarray[bool]
            has_switch   : np.ndarray[bool]
        """
        # Serial loop — SimAlign does not support batch inference here.
        # At ~27 pairs/s on GPU, expect ~60 min for 100k pairs.
        results = [self.score_pair(s, t) for s, t in zip(sources, targets)]
        return dict(
            gender_score=np.array([r["gender_score"] for r in results]),
            score_A=np.array([r["score_A"] for r in results]),
            score_B=np.array([r["score_B"] for r in results]),
            activated=np.array([r["activated"] for r in results]),
            has_switch=np.array([r["has_switch"] for r in results]),
        )
