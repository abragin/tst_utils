"""
Entity Consistency Metric (Task 1.7).

Detects spurious entity substitutions between source and styled text.
An entity substitution occurs when the style-transfer model replaces a named
entity from the source (e.g. "Жилин") with one not present in the source
(e.g. "Маруся", a character typical of the target author).

Score convention: 1.0 = no suspicious new entities, 0.0 = all target entities
are new and unmatched. A pair is flagged when entity_score < threshold or
when has_substitution is True (n_suspicious_new >= 1).

Method
------
Three complementary approaches:

  Approach A (entity-set comparison, primary):
    1. Extract named entities from source and target using Gherman NER
       (active types: FIRST_NAME, LAST_NAME, MIDDLE_NAME, CITY, REGION, COUNTRY).
    2. Optionally strip BI tokens from target with BiClEnsemble before NER —
       boilerplate often contains target-author character names.
    3. For each target entity, find the best string-similarity match among
       source entities of the same type (SequenceMatcher ratio). If
       max_sim >= string_sim_threshold (default 0.75), it is a morphological
       variant — suppress. Remaining target-only entities are "suspicious".
    4. entity_score = 1 - n_suspicious / max(n_tgt_entities, 1).
       has_substitution = n_suspicious >= 1.

  Approach C (entity-bag LaBSE, supplementary):
    Concatenate all NER-tagged words from source → embed with LaBSE. Same for
    target. entity_bag_sim = cosine similarity. Captures global entity-set
    shift without entity-pair matching. Returned only when use_labse=True.
    Edge case: if one side has no entities, bag_sim = 0.0 (entity appeared
    from nothing or vanished). If both sides have no entities, bag_sim = 1.0.

  Approach D (NE-filtered BERTScore):
    Run a BERT encoder over both full texts. For each NE-tagged token position,
    compute per-token BERTScore against ALL tokens of the other text:
      - ne_bert_precision: for each NE token in target, max cosine sim to any
        source token, averaged. Low → target introduced entity not in source.
      - ne_bert_recall: for each NE token in source, max cosine sim to any
        target token, averaged. Low → source entity has no match in target.
      - ne_bert_f1: harmonic mean of the above.
    Advantages over Approach A:
      - Morphological variants handled natively by BERT embeddings.
      - NER misses on one side don't cause FP: a missed entity word is still
        present as a token in the full text and contributes to similarity.
      - Graded score, no string-similarity threshold needed.
    Returned only when use_bert_score=True.
    Edge cases: no NE tokens on a side → that side's score = 1.0 (vacuous).

Known blind spots:
  - Rivers, waterways, mountains, organizations (Gherman NER has no such types).
  - Role-word substitutions ("князь" → "президент") — common nouns, not tagged.
  - Semantic aliases with different surface forms ("Ленинград" → "Питер"):
    neither string similarity nor LaBSE reliably suppresses these. Accepted as
    known false-positive category. Approach D may handle these better via
    contextual BERT similarity.

Validation (llm_analysis_combined.csv, n=451 after ground-truth filtering, 22.6% prevalence):
  A alone AUC=0.822; mean(A,C)=0.841; mean(A,D)=0.837 (val set).
  LLM validation on 168-row train_short sample (κ=0.844): A AUC=0.756, combined AUC=0.784.
  Binary threshold (entity_score < 1.0): P=0.341 R=0.848.
  Binary threshold (combined_entity_score < 1.0): P=0.264 R=1.000 (zero FNs).
  See notebook 09 entity consistency analysis.ipynb for full results.

Dependencies: transformers (Gherman NER), difflib (stdlib), and optionally
  sentence_transformers (LaBSE), tst_utils.eval.bi_cl_cleaner (BiClEnsemble),
  transformers AutoModel (Approach D BERTScore).
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

# Entity types active for substitution detection.
# STREET/DISTRICT/HOUSE excluded — noisy for literary text.
PERSON_TYPES = frozenset({"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"})
LOCATION_TYPES = frozenset({"CITY", "REGION", "COUNTRY"})
ACTIVE_TYPES = PERSON_TYPES | LOCATION_TYPES

# All person-name entity types including the merged "PERSON" type produced
# when consecutive FIRST/MIDDLE/LAST tokens are joined into one span.
_ALL_PERSON_TYPES = PERSON_TYPES | {"PERSON"}


def _string_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio between lowercased strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_string_sim(word: str, candidates: List[str]) -> float:
    """Max string similarity between word and any candidate. 0.0 if no candidates."""
    if not candidates:
        return 0.0
    return max(_string_sim(word, c) for c in candidates)


class EntityConsistencyScorer:
    """
    Entity substitution detector.

    Loads models sequentially — NER on init, BiClEnsemble and/or LaBSE and/or
    BERTScorer lazily on first use (controlled by pre_clean, use_labse, and
    use_bert_score flags).

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'.
    pre_clean : bool
        If True, run BiClEnsemble.score_and_clean_pair(use_hybrid=True) on the
        target before NER to strip boilerplate injection tokens. Requires
        bi_cl_model_path.
    bi_cl_model_path : str or None
        Path to the BiClEnsemble model directory (e.g.
        '15 metrics exploration/data/bi_cl_model/best'). Required when
        pre_clean=True.
    use_labse : bool
        If True, also compute entity_bag_sim (Approach C) via LaBSE.
    use_bert_score : bool
        If True, also compute ne_bert_precision / ne_bert_recall / ne_bert_f1
        (Approach D) using a BERT encoder. See bert_score_model.
    bert_score_model : str or None
        HuggingFace model name for Approach D embeddings.
        If None (default), reuses the NER model's base BERT encoder (already
        on GPU, no extra memory). Pass a model name to load a separate model —
        it must use safetensors weights (torch.load restriction in torch < 2.6).
    string_sim_threshold : float
        Morphological variant suppression threshold for Approach A. Default 0.75.

    Examples
    --------
    >>> scorer = EntityConsistencyScorer(pre_clean=False)
    >>> r = scorer.score_pair("Жилин не понял старика.", "Маруся не поняла старика.")
    >>> r['has_substitution']   # True — "Маруся" is new
    True
    >>> r = scorer.score_pair("Он жил в Ленинграде.", "Он жил в Ленинграду.")
    >>> r['has_substitution']   # False — morphological variant
    False
    """

    def __init__(
        self,
        device: str = "cuda",
        pre_clean: bool = False,
        bi_cl_model_path: Optional[str] = None,
        use_labse: bool = False,
        use_bert_score: bool = False,
        bert_score_model: Optional[str] = None,
        string_sim_threshold: float = 0.75,
    ):
        self.device = device
        self.pre_clean = pre_clean
        self.bi_cl_model_path = bi_cl_model_path
        self.use_labse = use_labse
        self.use_bert_score = use_bert_score
        self.bert_score_model = bert_score_model
        self.string_sim_threshold = string_sim_threshold

        self._ner = self._load_ner(device)
        self._ensemble = None   # lazy — loaded on first score_pair if pre_clean=True
        self._labse = None      # lazy — loaded on first score_pair if use_labse=True
        self._bert_scorer = None  # lazy — loaded on first score_pair if use_bert_score=True

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_ner(device: str):
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers is required: pip install transformers")
        device_id = 0 if device == "cuda" else -1
        return pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            aggregation_strategy="simple",
            device=device_id,
        )

    def _truncate_to_tokens(self, text: str, max_tokens: int = 500) -> str:
        """Truncate text at the character boundary corresponding to max_tokens BERT tokens.

        Ensures NER and Approach D BERTScore see the same text window — the old
        text[:512] (character) limit was far more conservative (~80–100 tokens)
        than the 512-token limit the BERT encoder accepts.
        """
        tokenizer = self._ner.tokenizer
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )
        if len(enc["input_ids"]) < max_tokens:
            return text
        return text[: enc["offset_mapping"][-1][1]]

    def _ensure_ensemble(self):
        if self._ensemble is not None:
            return
        if not self.bi_cl_model_path:
            raise ValueError(
                "pre_clean=True requires bi_cl_model_path to be set. "
                "Pass the path to the BiClEnsemble model directory, e.g. "
                "'15 metrics exploration/data/bi_cl_model/best'."
            )
        try:
            from tst_utils.eval.bi_cl_cleaner import BiClEnsemble
        except ImportError:
            raise ImportError("tst_utils.eval.bi_cl_cleaner is required for pre_clean=True.")
        self._ensemble = BiClEnsemble(model_path=self.bi_cl_model_path, device=self.device)
        LOG.info("BiClEnsemble loaded from %s", self.bi_cl_model_path)

    def _ensure_labse(self):
        if self._labse is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence_transformers is required for use_labse=True.")
        self._labse = SentenceTransformer("cointegrated/LaBSE-en-ru", device=self.device)
        LOG.info("LaBSE loaded")

    def _ensure_bert_scorer(self):
        if self._bert_scorer is not None:
            return
        if self.bert_score_model is None:
            # Reuse the NER model's base BERT encoder — already on GPU, no extra memory.
            # Gherman NER is BertForTokenClassification; its base encoder is .bert
            tokenizer = self._ner.tokenizer
            model = self._ner.model.bert
            self._bert_scorer = (tokenizer, model)
            LOG.info("BERTScorer: reusing NER model encoder (Gherman .bert)")
        else:
            try:
                from transformers import AutoTokenizer, AutoModel
            except ImportError:
                raise ImportError("transformers is required for use_bert_score=True.")
            import torch
            tokenizer = AutoTokenizer.from_pretrained(self.bert_score_model)
            model = AutoModel.from_pretrained(self.bert_score_model)
            model = model.to(self.device).eval()
            self._bert_scorer = (tokenizer, model)
            LOG.info("BERTScorer loaded: %s", self.bert_score_model)

    def _parse_ner(self, text: str) -> Tuple[Dict[str, List[str]], List[Tuple[int, int]]]:
        """
        Run NER, merge consecutive spans, return (entities_dict, char_spans).

        char_spans is a list of (start, end) character offsets in text,
        one entry per merged entity span. Parallel to the entity words in
        entities_dict (same order as the merged list).

        Merging rules:
          - GEO (CITY/REGION/COUNTRY): same-type adjacent spans joined into one,
            e.g. "Российской"(COUNTRY) + "Федерации"(COUNTRY) → "Российской Федерации".
          - Person names: any adjacent FIRST/MIDDLE/LAST sequence joined into a
            single "PERSON" span. Avoids split-name false positives.

        Returns
        -------
        entities_dict : dict mapping entity_type → list of entity surface forms.
        char_spans : list of (start, end) for each merged entity span.
        """
        raw = self._ner(self._truncate_to_tokens(text))

        # Filter: active types only, drop subword artifacts
        valid = [e for e in raw
                 if e["entity_group"] in ACTIVE_TYPES
                 and not e["word"].startswith("##")
                 and len(e["word"]) >= 2]

        if not valid:
            return {}, []

        # Merge consecutive adjacent spans (gap ≤ 2 chars = one space)
        merged = [dict(valid[0])]
        for e in valid[1:]:
            prev = merged[-1]
            gap = e["start"] - prev["end"]
            prev_grp = prev["entity_group"]
            cur_grp = e["entity_group"]

            if gap <= 2:
                # Same-type GEO: join into one span
                if prev_grp == cur_grp and prev_grp in LOCATION_TYPES:
                    prev["word"] = prev["word"] + " " + e["word"]
                    prev["end"] = e["end"]
                    continue
                # Any PERSON-type sequence: join into unified "PERSON" span
                if prev_grp in _ALL_PERSON_TYPES and cur_grp in PERSON_TYPES:
                    prev["word"] = prev["word"] + " " + e["word"]
                    prev["entity_group"] = "PERSON"
                    prev["end"] = e["end"]
                    continue

            merged.append(dict(e))

        entities: Dict[str, List[str]] = {}
        spans: List[Tuple[int, int]] = []
        for e in merged:
            entities.setdefault(e["entity_group"], []).append(e["word"])
            spans.append((e["start"], e["end"]))

        return entities, spans

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Run NER and return active entities grouped by type."""
        entities, _ = self._parse_ner(text)
        return entities

    def _find_suspicious(
        self,
        src_ents: Dict[str, List[str]],
        tgt_ents: Dict[str, List[str]],
    ) -> List[Tuple[str, str]]:
        """
        Return list of (entity_type, entity_word) for target entities that have
        no close string match in the source.

        Matching rules:
          - Person types (FIRST_NAME, LAST_NAME, MIDDLE_NAME, PERSON): compared
            against ALL person-type source words regardless of subtype. This
            handles the case where merging produced "PERSON" on one side but
            individual subtypes on the other.
          - Location types (CITY, REGION, COUNTRY): same-type comparison only.
        """
        # Collect all source person words across subtypes.
        # For merged multi-word PERSON spans (e.g. "Иван Петрович"), also add
        # the individual component words so that a target referencing only part
        # of the name (e.g. "Ивану", dative of "Иван") can still match.
        src_person_words = []
        for t in _ALL_PERSON_TYPES:
            for w in src_ents.get(t, []):
                src_person_words.append(w)
                if " " in w:
                    src_person_words.extend(w.split())

        suspicious = []
        for etype, tgt_words in tgt_ents.items():
            src_words = src_person_words if etype in _ALL_PERSON_TYPES else src_ents.get(etype, [])
            for tw in tgt_words:
                sim = _best_string_sim(tw, src_words)
                if sim < self.string_sim_threshold:
                    suspicious.append((etype, tw))
        return suspicious

    def _entity_bag_sim(
        self,
        src_ents: Dict[str, List[str]],
        tgt_ents: Dict[str, List[str]],
    ) -> float:
        """
        LaBSE cosine similarity between entity bags (Approach C).

        Concatenate all NER-tagged words from each side and embed with LaBSE.
        Returns:
          1.0 if both sides have no entities (no change to compare).
          0.0 if exactly one side has no entities (entities appeared or vanished).
          cosine similarity otherwise.
        """
        src_bag = " ".join(w for words in src_ents.values() for w in words)
        tgt_bag = " ".join(w for words in tgt_ents.values() for w in words)
        if not src_bag and not tgt_bag:
            return 1.0
        if not src_bag or not tgt_bag:
            return 0.0
        embs = self._labse.encode([src_bag, tgt_bag], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def _ne_bert_score(
        self,
        src_text: str,
        tgt_text: str,
        src_spans: List[Tuple[int, int]],
        tgt_spans: List[Tuple[int, int]],
    ) -> Tuple[float, float, float]:
        """
        NE-filtered BERTScore (Approach D).

        Encode both texts with a BERT model to get per-token contextual
        embeddings. Identify which token positions overlap NE character spans.
        Compute BERTScore-style precision and recall restricted to NE tokens,
        matched against ALL tokens of the other text (not just NE tokens).

        Matching against all tokens (not NE-only) means a missed NE on one
        side still participates in similarity via its full-text token embedding,
        avoiding FPs from NER case-form misses.

        Empty NE token set on a side → that side's score = 1.0 (vacuous):
          - No target NEs → precision = 1.0 (no new entities introduced)
          - No source NEs → recall = 1.0 (nothing to preserve)

        Returns
        -------
        (ne_bert_precision, ne_bert_recall, ne_bert_f1)
        """
        import torch
        import torch.nn.functional as F

        # Truncate both texts to the same token boundary used by _parse_ner so
        # that NER character spans are valid within the text the encoder sees.
        src_text = self._truncate_to_tokens(src_text)
        tgt_text = self._truncate_to_tokens(tgt_text)

        tokenizer, model = self._bert_scorer

        def _encode(text: str, ne_spans: List[Tuple[int, int]]):
            """Tokenize, encode, return (normalized_embeddings, ne_token_mask)."""
            enc = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=512,
            )
            offsets = enc.pop("offset_mapping")[0]  # (n_tok, 2)
            with torch.no_grad():
                hidden = model(
                    **{k: v.to(self.device) for k, v in enc.items()}
                ).last_hidden_state[0]  # (n_tok, H)
            embs = F.normalize(hidden, dim=-1).cpu()

            # NE mask: token overlaps any NE character span
            ne_mask = torch.zeros(len(offsets), dtype=torch.bool)
            for s, e in ne_spans:
                overlap = (offsets[:, 1] > s) & (offsets[:, 0] < e)
                ne_mask |= overlap

            # Exclude [CLS] / [SEP] (offset_mapping = (0, 0) for special tokens)
            special = (offsets[:, 0] == 0) & (offsets[:, 1] == 0)
            ne_mask &= ~special

            return embs, ne_mask

        src_embs, src_ne_mask = _encode(src_text, src_spans)
        tgt_embs, tgt_ne_mask = _encode(tgt_text, tgt_spans)

        tgt_ne_embs = tgt_embs[tgt_ne_mask]  # (n_tgt_ne, H)
        src_ne_embs = src_embs[src_ne_mask]  # (n_src_ne, H)

        # Precision: each target NE token vs ALL source tokens
        if len(tgt_ne_embs) == 0:
            precision = 1.0
        else:
            # (n_tgt_ne, n_src) → max over source axis → mean over target NEs
            # Clip to [0, 1]: cosine sim can be slightly negative for the NER
            # model's embedding space; negative = no match, so 0 is the right floor.
            sim = (tgt_ne_embs @ src_embs.T).max(dim=1).values.clamp(min=0.0)
            precision = float(sim.mean())

        # Recall: each source NE token vs ALL target tokens
        if len(src_ne_embs) == 0:
            recall = 1.0
        else:
            sim = (src_ne_embs @ tgt_embs.T).max(dim=1).values.clamp(min=0.0)
            recall = float(sim.mean())

        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return precision, recall, f1

    # ── public interface ──────────────────────────────────────────────────────

    def score_pair(self, source: str, target: str) -> dict:
        """
        Score a single (source, target) pair.

        Parameters
        ----------
        source : str
            Original source text.
        target : str
            Styled paraphrase. If pre_clean=True, BI tokens are stripped
            from this text before NER.

        Returns
        -------
        dict with:
            entity_score        : float  — 1.0=no suspicious substitution, 0.0=all
                                           target entities are new/unmatched.
            has_substitution    : bool   — True if at least one suspicious new entity.
            new_entities        : list   — [(entity_type, entity_word), ...] for each
                                           suspicious new target entity.
            src_entities        : dict   — {type: [words]} extracted from source.
            tgt_entities        : dict   — {type: [words]} extracted from target
                                           (after BI cleaning if pre_clean=True).
            cleaned_target      : str    — BI-cleaned target (only if pre_clean=True).
            entity_bag_sim      : float  — LaBSE entity-bag similarity (Approach C,
                                           only if use_labse=True).
            ne_bert_precision   : float  — NE-token BERTScore precision (Approach D,
                                           only if use_bert_score=True).
            ne_bert_recall      : float  — NE-token BERTScore recall (Approach D).
            ne_bert_f1          : float  — NE-token BERTScore F1 (Approach D).
            combined_entity_score : float — min(entity_score, ne_bert_precision);
                                           recommended single signal when use_bert_score=True.
                                           Equivalent to max(1−entity_score, 1−ne_bert_precision)
                                           as a suspicion score. AUC=0.784 vs A alone 0.756
                                           on 168-row train_short LLM sample. Binary threshold
                                           (< 1.0) gives R=1.000 with 0 FNs; A alone gives
                                           R=0.848. Uses D precision (not F1): recall captures
                                           entity dropping, a different failure mode.
        """
        if not source.strip() or not target.strip():
            result = dict(
                entity_score=1.0,
                has_substitution=False,
                new_entities=[],
                src_entities={},
                tgt_entities={},
            )
            if self.use_labse:
                result["entity_bag_sim"] = 1.0
            if self.use_bert_score:
                result["ne_bert_precision"] = 1.0
                result["ne_bert_recall"] = 1.0
                result["ne_bert_f1"] = 1.0
                result["combined_entity_score"] = 1.0
            return result

        # Optional BI pre-cleaning
        tgt_for_ner = target
        cleaned_target = None
        if self.pre_clean:
            self._ensure_ensemble()
            clean_result = self._ensemble.score_and_clean_pair(source, target, use_hybrid=True)
            tgt_for_ner = clean_result["cleaned_target"]
            cleaned_target = tgt_for_ner

        # NER extraction — always get spans for potential use in Approach D
        src_ents, src_spans = self._parse_ner(source)
        tgt_ents, tgt_spans = self._parse_ner(tgt_for_ner)

        # Entity-set comparison (Approach A)
        suspicious = self._find_suspicious(src_ents, tgt_ents)
        n_suspicious = len(suspicious)
        n_tgt = sum(len(v) for v in tgt_ents.values())
        entity_score = max(0.0, 1.0 - n_suspicious / max(n_tgt, 1))

        result = dict(
            entity_score=entity_score,
            has_substitution=n_suspicious >= 1,
            new_entities=suspicious,
            src_entities=src_ents,
            tgt_entities=tgt_ents,
        )
        if cleaned_target is not None:
            result["cleaned_target"] = cleaned_target

        # Entity-bag LaBSE (Approach C)
        if self.use_labse:
            self._ensure_labse()
            result["entity_bag_sim"] = self._entity_bag_sim(src_ents, tgt_ents)

        # NE-filtered BERTScore (Approach D)
        if self.use_bert_score:
            self._ensure_bert_scorer()
            p, r, f1 = self._ne_bert_score(source, tgt_for_ner, src_spans, tgt_spans)
            result["ne_bert_precision"] = p
            result["ne_bert_recall"] = r
            result["ne_bert_f1"] = f1
            # Combined signal: A precision (string-match) AND D precision (embedding).
            # Uses D precision rather than F1 — D recall (source entity dropped)
            # is a different failure mode and adds noise for substitution detection.
            result["combined_entity_score"] = min(entity_score, p)

        return result

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
            entity_score        : np.ndarray[float]
            has_substitution    : np.ndarray[bool]
            new_entities        : list of lists (not an array — variable length)
            entity_bag_sim      : np.ndarray[float]  (only if use_labse=True)
            ne_bert_precision   : np.ndarray[float]  (only if use_bert_score=True)
            ne_bert_recall      : np.ndarray[float]  (only if use_bert_score=True)
            ne_bert_f1          : np.ndarray[float]  (only if use_bert_score=True)
        """
        results = [self.score_pair(s, t) for s, t in zip(sources, targets)]
        out = dict(
            entity_score=np.array([r["entity_score"] for r in results]),
            has_substitution=np.array([r["has_substitution"] for r in results]),
            new_entities=[r["new_entities"] for r in results],
        )
        if self.use_labse:
            out["entity_bag_sim"] = np.array([r["entity_bag_sim"] for r in results])
        if self.use_bert_score:
            out["ne_bert_precision"] = np.array([r["ne_bert_precision"] for r in results])
            out["ne_bert_recall"] = np.array([r["ne_bert_recall"] for r in results])
            out["ne_bert_f1"] = np.array([r["ne_bert_f1"] for r in results])
            out["combined_entity_score"] = np.array([r["combined_entity_score"] for r in results])
        return out
