"""``TstPerformanceMetrics`` — orchestrator for the three-pillar TST evaluation.

Thin orchestrator: reshaping (``reshape``), source caching (``source_cache``),
and the per-metric base scores (``scoring``) live in sibling submodules; this
module sequences them and adds the composite/quality/selection logic.
"""

import logging
import warnings

import pandas as pd
import numpy as np
from tst_utils.eval.metrics.meaning import meaning_score
from tst_utils.eval.metrics.composite import base_score_v2, compute_nat_v2
from tst_utils.eval import model_names
from tst_utils.eval.data.style_store import (
    LabelledStyleVectors,
    StyleLabelError,
    load_style_embeddings,
    style_text_key,
    warn_provenance,
)
from tst_utils.eval.style_encoder_registry import get_encoder
from tst_utils.eval.performance.constants import TARGET_STYLES, _SCORE_COLS, _QUALITY_COLS
from tst_utils.eval.performance.reshape import expand_tst_output
from tst_utils.eval.performance.source_cache import (
    encode_style_rows, ensure_source_caches)
from tst_utils.eval.performance.copying import add_chrf
from tst_utils.eval.performance import scoring

logger = logging.getLogger(__name__)


def _same_vector(cell, stored):
    """True when a frame cell holds exactly the stored vector."""
    try:
        array = np.asarray(cell)
    except Exception:
        return False
    return array.shape == stored.shape and np.array_equal(array, stored)


# The score_v2 stage's outputs. They depend on style_score, and
# compute_scores does not recompute them, so it drops them (F2 of the results
# review of task 5.2); compute_composite_v2 recomputes them.
_STYLE_DERIVED_COLS = ["meaning_nolen", "nat_v2", "score_v2"]


def _is_null(value):
    """True for a missing scalar cell (None or NaN)."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class TstPerformanceMetrics:
    """
    Class to evaluate TST models using style transfer, meaning, and naturality metrics.

    Parameters:
    - test_df (pd.DataFrame): Dataset containing source texts and related metadata.
        Expected columns:
          • For TST with string target styles: 'text' and 'author'.
          • For TST with explicit target style vectors (the emb path):
              'text', 'author', 'target_style_key' and 'target_style_desc'.
              `target_style_key` selects each row's target from
              `target_style_source`; `target_style_desc` is free text and
              never read as a key.
    - tst_func (Callable): Function performing text style transfer.
        • Text path: (texts: List[str], target_style: str) -> List[str].
        • Emb path: (texts=List[str], target_style_embeddings=StyleSelection)
          -> List[str]. The selection carries the labelled source and one
          row key per text, never bare vectors.
    - target_styles (List[str] | None): target styles of the text path; None
        selects the emb path.
    - tst_model (str): Name or ID of the TST model being evaluated.
    - author_styles (LabelledStyleVectors | None): the scoring targets of the
        text path, keyed by target style, labelled with the evaluation
        encoder.
    - verbose (bool): Whether to log debug/progress information during processing.
    - evaluation_encoder (str | None): the registry key every style encode
        and every style check of this instance uses. None means the pin,
        read once here; it is never reread during a run.
    - target_style_source (LabelledStyleVectors | None): the targets of the
        emb path, labelled with the evaluation encoder.
    - source_style_path (str | None): a labelled parquet side file of source
        style vectors, checked against the evaluation encoder at every entry.

    Source style vectors are owned by the instance (task 5.2, requirement
    11): a private store maps the exact text key of each source text to a
    write-protected vector, filled only by the instance's own encode and by
    a checked side-file load. The `text_style_emb` column is a detached
    copy written from the store; a cell that differs from the store, or a
    text the store has never seen, is warned about, discarded and encoded
    fresh.

    While the pin and the evaluation encoder are both `base_v1`, a bare
    dict `author_styles` and an inline `target_style_emb` column are
    accepted with a warning, as declared `base_v1` sources (the J4-tested
    tree and a rollback keep `PphGenerator` working). Otherwise both raise.
    """
    def __init__(
        self, test_df, tst_func, target_styles, tst_model,
        author_styles, verbose = False, perplexity_batch_size = 32, *,
        evaluation_encoder=None, target_style_source=None,
        source_style_path=None,
    ):
        default_pin = model_names.style_encoder_key()
        if evaluation_encoder is None:
            evaluation_encoder = default_pin
        get_encoder(evaluation_encoder)  # an unregistered key raises here
        self._evaluation_encoder = evaluation_encoder
        # The legacy intake (bare dict, inline target column) is open only
        # while both the default pin and this run are base_v1.
        self._legacy_v1 = (default_pin == "base_v1"
                           and evaluation_encoder == "base_v1")
        self._default_pin = default_pin
        self.test_df = test_df
        self.tst_func = tst_func
        self.target_styles = target_styles
        self.tst_model = tst_model
        self.verbose = verbose
        # Both setters run the constructor's intake, so a later assignment
        # is checked as strictly as the constructor argument.
        self.author_styles = author_styles
        self.target_style_source = target_style_source
        self._inline_target_source = None
        self.source_style_path = source_style_path
        self._source_store = {}
        # rugpt3 perplexity batch size; lower to fit a small GPU (the per-batch
        # logits tensor is the VRAM bottleneck — see styled_pph_gen's 8 GB path).
        self.perplexity_batch_size = perplexity_batch_size
        self.tst_results = None
        if verbose and not logger.handlers:
            # Verbose progress used to go to stdout via print(); preserve that it
            # is actually visible even when the host app hasn't configured logging
            # (a bare logger.info would be dropped by the WARNING last-resort handler).
            _handler = logging.StreamHandler()
            _handler.setFormatter(
                logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
            logger.addHandler(_handler)
            logger.setLevel(logging.INFO)

    @property
    def evaluation_encoder(self):
        """The registry key of this run's evaluation encoder (read-only)."""
        return self._evaluation_encoder

    @property
    def author_styles(self):
        """The labelled target source of the text path (named targets)."""
        return self._author_styles

    @author_styles.setter
    def author_styles(self, targets):
        self._author_styles = self._intake_targets(targets, "author_styles")

    @property
    def target_style_source(self):
        """The labelled target source of the emb path (key-selected)."""
        return self._target_style_source

    @target_style_source.setter
    def target_style_source(self, targets):
        self._target_style_source = self._intake_targets(
            targets, "target_style_source")

    # ---- intake and validation (task 5.2) ---------------------------------

    def _intake_targets(self, targets, name):
        """Accept a labelled target source of the evaluation encoder."""
        if targets is None or isinstance(targets, LabelledStyleVectors):
            if (targets is not None
                    and targets.encoder_key != self._evaluation_encoder):
                raise StyleLabelError(
                    f"TstPerformanceMetrics: {name} holds "
                    f"{targets.encoder_key!r} vectors; the evaluation "
                    f"encoder is {self._evaluation_encoder!r}")
            return targets
        if isinstance(targets, dict):
            if not self._legacy_v1:
                raise StyleLabelError(
                    f"TstPerformanceMetrics: {name} is a bare dict; pass "
                    "LabelledStyleVectors (load_labelled_author_styles, "
                    "from_build, from_encoded or declare); the pin is "
                    f"{self._default_pin!r} and the evaluation encoder "
                    f"{self._evaluation_encoder!r}")
            warn_provenance(
                f"TstPerformanceMetrics: {name} is a bare dict with no "
                "label; declared base_v1 while the pin is base_v1")
            keys = [k for k in targets if not str(k).startswith("__")]
            return LabelledStyleVectors.declare(
                {k: targets[k] for k in keys}, "base_v1",
                f"bare dict {name} (legacy, pin base_v1)", keys=keys)
        raise TypeError(
            f"TstPerformanceMetrics: {name} must be LabelledStyleVectors, "
            f"got {type(targets).__name__}")

    def _put_source(self, key, vector):
        stored = np.array(vector, copy=True)
        stored.setflags(write=False)
        self._source_store[key] = stored

    def _side_file_vectors(self):
        """The checked side file as {text key: vector}; raises on a mismatch."""
        frame, _label = load_style_embeddings(
            self.source_style_path, expect_encoder=self._evaluation_encoder)
        return {key: np.asarray(vector, dtype=np.float32)
                for key, vector in zip(frame["text_key"], frame["style_emb"])}

    def _validate_source_style(self, frame, entry_point):
        """Resolve every source text against the store; return the store's
        vectors in row order and the frame with a detached `text_style_emb`.
        """
        texts = frame["text"].tolist()
        keys = [style_text_key(text) for text in texts]
        side = (self._side_file_vectors()
                if self.source_style_path is not None else None)
        if side is not None:
            missing = [text for key, text in zip(keys, texts)
                       if key not in side]
            if missing:
                raise StyleLabelError(
                    f"{entry_point}: {len(missing)} texts have no row in "
                    f"{self.source_style_path}, first missing excerpt: "
                    f"{str(missing[0])[:80]!r}")
        affected = [False] * len(keys)
        if "text_style_emb" in frame.columns:
            for row, (key, cell) in enumerate(zip(keys,
                                                  frame["text_style_emb"])):
                stored = self._source_store.get(key)
                affected[row] = stored is None or not _same_vector(cell, stored)
            if any(affected):
                warn_provenance(
                    f"{entry_point}: stored column text_style_emb has no "
                    "label or differs from the instance's source store; "
                    "the affected rows are dropped and encoded fresh")
        need, seen = [], set()
        for key, text, bad in zip(keys, texts, affected):
            if (bad or key not in self._source_store) and key not in seen:
                seen.add(key)
                need.append((key, text))
        if need:
            if side is not None:
                for key, _text in need:
                    self._put_source(key, side[key])
            else:
                vectors = encode_style_rows([text for _key, text in need],
                                            self._evaluation_encoder)
                for (key, _text), vector in zip(need, vectors):
                    self._put_source(key, vector)
        frame = frame.copy()
        source_vectors = [self._source_store[key] for key in keys]
        frame["text_style_emb"] = [np.array(v, copy=True)
                                   for v in source_vectors]
        return frame, source_vectors

    def _emb_mode(self):
        return not self.target_styles

    def _emb_targets(self):
        return self.target_style_source or self._inline_target_source

    def _scoring_targets(self):
        """The labelled target source that scores this instance's mode."""
        return self._emb_targets() if self._emb_mode() else self.author_styles

    def _check_target_encoder(self, targets, entry_point):
        """Re-check a target source against the evaluation encoder at use."""
        if targets is not None and targets.encoder_key != self._evaluation_encoder:
            raise StyleLabelError(
                f"{entry_point}: the target source holds "
                f"{targets.encoder_key!r} vectors; the evaluation encoder is "
                f"{self._evaluation_encoder!r}")

    def _validate_targets(self, frame, entry_point, require_desc=True):
        """Check the target supply of `frame` before any encode or TST call.

        On the emb path the frame must carry `target_style_key` (after the
        legacy inline conversion), and a `test_df` entry also
        `target_style_desc`; `compute_scores` passes `require_desc=False`,
        because scoring an external result frame needs no description.
        """
        if not self._emb_mode():
            if "target_style_emb" in frame.columns:
                self._reject_or_warn_inline(entry_point)
            self._check_target_encoder(self.author_styles, entry_point)
            return frame
        frame = frame.copy()
        if "target_style_emb" in frame.columns:
            self._reject_or_warn_inline(entry_point)
            if ("target_style_key" not in frame.columns
                    and self.target_style_source is None):
                # Legacy: the inline column becomes a declared base_v1 source,
                # one key per source row.
                if self._inline_target_source is None:
                    keys = [f"inline_{i}" for i in range(len(frame))]
                    self._inline_target_source = LabelledStyleVectors.declare(
                        list(frame["target_style_emb"]), "base_v1",
                        f"inline target_style_emb at {entry_point} "
                        "(legacy, pin base_v1)", keys=keys)
                    frame["target_style_key"] = keys
        missing = [c for c in ("target_style_key", "target_style_desc")
                   if c not in frame.columns
                   and (require_desc or c == "target_style_key")]
        if missing:
            raise ValueError(
                f"{entry_point}: the emb path (no target_styles) needs the "
                f"column(s) {missing}; target_style_key selects each row's "
                "target from target_style_source")
        if "target_style_key" in frame.columns:
            source = self._emb_targets()
            if source is None:
                raise ValueError(
                    f"{entry_point}: target_style_key needs "
                    "target_style_source (LabelledStyleVectors)")
            self._check_target_encoder(source, entry_point)
            keys = frame["target_style_key"].tolist()
            bad = [k for k in keys if not isinstance(k, str)]
            if bad:
                raise ValueError(f"{entry_point}: target_style_key must be "
                                 f"a str on every row, got {bad[0]!r}")
            unknown = sorted({k for k in keys if k not in source})
            if unknown:
                raise KeyError(f"{entry_point}: unknown target_style_key "
                               f"values {unknown}; known: "
                               f"{list(source.keys)}")
        return frame

    def _reject_or_warn_inline(self, entry_point):
        if not self._legacy_v1:
            raise StyleLabelError(
                f"{entry_point}: stored column target_style_emb has no "
                f"label; the pin is {self._default_pin!r} and the "
                f"evaluation encoder {self._evaluation_encoder!r}; select "
                "targets by target_style_key from a labelled source")
        warn_provenance(
            f"{entry_point}: stored column target_style_emb has no label; "
            "an unlabelled file predates the label")

    def _validate_frame(self, frame, entry_point, require_desc=True):
        """The shared validation of every public entry (requirement 11)."""
        if "text" not in frame.columns:
            raise ValueError(f"{entry_point}: the frame has no text column")
        frame = self._validate_targets(frame, entry_point, require_desc)
        frame, source_vectors = self._validate_source_style(frame,
                                                            entry_point)
        return frame, source_vectors

    def _target_record(self):
        """The result columns that record this run's targets and encoder."""
        source = self._scoring_targets()
        record = {"evaluation_encoder": self._evaluation_encoder}
        if source is not None:
            record.update({"target_provenance": source.provenance,
                           "target_declared": bool(source.is_declared),
                           "target_source": source.source})
        return record

    def _stamp_target_record(self, frame, entry_point):
        """Write this instance's record onto every row of `frame`.

        The record describes the scoring this instance does. A non-null
        value that differs (for example a frame scored earlier by another
        instance) is replaced, with a provenance warning naming both values;
        an absent column or a null cell is filled silently.
        """
        frame = frame.copy()
        for column, value in self._target_record().items():
            if column in frame.columns:
                old = sorted({str(v) for v in frame[column].tolist()
                              if not _is_null(v) and v != value})
                if old:
                    warn_provenance(
                        f"{entry_point}: tst_results column {column} held "
                        f"{old}; replaced by {value!r}, the record of this "
                        "instance's scoring")
            frame[column] = [value] * len(frame)
        return frame

    def _log(self, msg, *args) -> None:
        """Verbose-gated progress logging (replaces the old `if self.verbose: print`)."""
        if self.verbose:
            logger.info(msg, *args)

    def produce_tst_results_with_text_target_styles(self) -> None:
        entry = "TstPerformanceMetrics.produce_tst_results_with_text_target_styles"
        self.test_df, _ = self._validate_frame(self.test_df, entry)
        res_dfs = []

        precalculated_cols = ["text_perplexity", "text_style_emb", "text_labse_emb"]
        cols_to_copy = (
            ['example_number', 'text', 'author'] +
            [c for c in precalculated_cols if c in self.test_df]
        )
        record = self._target_record()

        for t_style in self.target_styles:
            self._log("Performing TST for style: %s", t_style)

            df = self.test_df[self.test_df.author != t_style].copy()
            df = df.reset_index(drop=False).rename(columns={"index": "example_number"})
            df["target_style"] = t_style
            df["tst_model"] = self.tst_model

            # Run TST
            tst_output = self.tst_func(df.text.tolist(), t_style)

            df_expanded = expand_tst_output(
                tst_output, df, cols_to_copy,
                extra_fields={"target_style": t_style, "tst_model": self.tst_model,
                              **record},
            )
            res_dfs.append(df_expanded)

        # Combine all target styles
        self.tst_results = pd.concat(res_dfs, ignore_index=True)

    def produce_tst_results_with_target_style_emb(self) -> None:
        """
        Perform TST with targets selected by key from the labelled
        `target_style_source`. Expects `target_style_key` and
        `target_style_desc` columns in test_df (an inline
        `target_style_emb` column is accepted only while the pin is base_v1,
        with a warning, as a declared base_v1 source).
        """
        entry = "TstPerformanceMetrics.produce_tst_results_with_target_style_emb"
        if "target_style_desc" not in self.test_df or not (
                {"target_style_key", "target_style_emb"} & set(self.test_df.columns)):
            raise ValueError("test_df must contain 'target_style_key' and "
                             "'target_style_desc' columns.")
        self.test_df, _ = self._validate_frame(self.test_df, entry)

        df = self.test_df.reset_index(drop=False).rename(columns={"index": "example_number"})
        df["tst_model"] = self.tst_model

        precalculated_cols = ["text_perplexity", "text_style_emb", "text_labse_emb"]
        # target_keys carries the concrete sampled target provenance (author/domain
        # identities) written by add_target_style_emb. Copied through conditionally
        # because callers that build test_df by hand may not set it; when present it
        # must survive this whitelist or it is silently dropped before best_tst_results.
        # target_style_emb is copied only on the legacy inline route.
        optional_provenance_cols = ["target_keys", "target_style_emb"]
        cols_to_copy = (
            ['example_number', 'text', 'author', 'target_style_key',
             'target_style_desc'] +
            [c for c in precalculated_cols if c in self.test_df] +
            [c for c in optional_provenance_cols if c in self.test_df]
        )

        self._log("Performing TST with target_style_embeddings")

        selection = self._emb_targets().select(df.target_style_key.tolist())
        tst_output = self.tst_func(
            texts=df.text.tolist(),
            target_style_embeddings=selection,
        )

        self.tst_results = expand_tst_output(
            tst_output, df, cols_to_copy,
            extra_fields={"tst_model": self.tst_model, **self._target_record()},
        )

    def produce_tst_results(self) -> None:
        """
        Automatically select between:
        - produce_tst_results_with_text_target_styles()
        - produce_tst_results_with_target_style_emb()
        depending on configuration.
        """
        if self.target_styles:
            return self.produce_tst_results_with_text_target_styles()
        elif "target_style_desc" in self.test_df.columns and (
                {"target_style_key", "target_style_emb"} & set(self.test_df.columns)):
            return self.produce_tst_results_with_target_style_emb()
        else:
            raise ValueError(
                "Cannot determine TST mode: either provide target_styles "
                "or ensure test_df has both 'target_style_key' and 'target_style_desc' columns."
            )

    def add_source_ppx_and_emb(self):
        entry = "TstPerformanceMetrics.add_source_ppx_and_emb"
        self.test_df, _ = self._validate_frame(self.test_df, entry)
        ensure_source_caches(
            self.test_df, perplexity=True, labse_emb=True,
            perplexity_batch_size=self.perplexity_batch_size,
        )

    def compute_scores(self):
        """Add base v1 scores to self.tst_results (style/meaning/naturality + score).

        Pipeline (each step caches/derives in place on self.tst_results):
            perplexity → bert → labse → style embeddings/away-towards → combine.
        The source style vectors come from the instance's store, the styled
        texts are encoded with the evaluation encoder, and each target is
        the unit view of the labelled source's stored vector.
        """
        if self.tst_results is None:
            raise Exception("TST results are absent.")
        entry = "TstPerformanceMetrics.compute_scores"
        targets = self._require_scoring_targets(entry)
        key_column = "target_style_key" if self._emb_mode() else "target_style"
        if key_column not in self.tst_results.columns:
            raise ValueError(f"{entry}: tst_results has no {key_column} column")
        unknown = sorted({str(k) for k in self.tst_results[key_column]
                          if k not in targets})
        if unknown:
            raise KeyError(f"{entry}: unknown {key_column} values {unknown}")
        # Score a candidate frame and publish it only on success: a scorer
        # that raises leaves tst_results as it was, old scores with their
        # old record, never old scores under this instance's record.
        candidate, source_vectors = self._validate_frame(
            self.tst_results, entry, require_desc=False)
        # Columns that depend on the style scores and that this method does
        # not recompute (the score_v2 stage) are dropped, so a pre-scored
        # frame cannot carry a stale composite under the new record.
        stale = [c for c in _STYLE_DERIVED_COLS if c in candidate.columns]
        candidate = candidate.drop(columns=stale)
        scoring.add_perplexity_scores(
            candidate, self._log, batch_size=self.perplexity_batch_size)
        scoring.add_bert_score(candidate, self._log)
        scoring.add_labse_score(candidate, self._log)
        scoring.add_labelled_style_scores(
            candidate, targets, key_column=key_column,
            source_vectors=source_vectors,
            encoder_key=self._evaluation_encoder, log=self._log)
        scoring.combine_scores(candidate, self._log)
        self.tst_results = self._stamp_target_record(candidate, entry)
        # a selection made from the earlier scores is no longer valid
        self.best_tst_results = None

    def _require_scoring_targets(self, entry_point):
        """The labelled scoring target source, checked; raises if absent."""
        targets = self._scoring_targets()
        if targets is None:
            raise ValueError(
                f"{entry_point}: no labelled target source "
                f"({'target_style_source' if self._emb_mode() else 'author_styles'})")
        self._check_target_encoder(targets, entry_point)
        return targets

    def select_best_tst_version(self, score_col: str = 'score') -> None:
        """For each (example_number, target_style) pair, select the row with the
        highest value of *score_col* and store the result in self.best_tst_results.

        Parameters
        ----------
        score_col : str
            Column to maximise. Use 'score' (default) for the v1 composite or
            'score_v2' for the v2 composite (requires compute_composite_v2() first).
        """
        if not hasattr(self, "tst_results") or self.tst_results is None:
            raise AttributeError("tst_results is missing. Run produce_tst_results() first.")

        if score_col not in self.tst_results.columns:
            raise ValueError(
                f"Column '{score_col}' not found in tst_results. "
                f"Run {'compute_scores()' if score_col == 'score' else 'compute_composite_v2()'} first."
            )

        self.tst_results[score_col] = pd.to_numeric(self.tst_results[score_col], errors="coerce")
        valid_df = self.tst_results.dropna(subset=[score_col])

        if valid_df.empty:
            raise ValueError(f"No valid '{score_col}' values found in tst_results.")

        if "target_style" in valid_df.columns:
            idx = valid_df.groupby(["example_number", "target_style"])[score_col].idxmax()
        else:
            idx = valid_df.groupby(["example_number"])[score_col].idxmax()
        self.best_tst_results = valid_df.loc[idx].reset_index(drop=True)

    def final_performance(self, score_col: str = 'score'):
        """Return mean scores by target style set from best_tst_results.

        Parameters
        ----------
        score_col : str
            Column to average. Use 'score' (default, v1) or 'score_v2'.
        """
        if getattr(self, "best_tst_results", None) is None:
            raise AttributeError(
                "best_tst_results is missing. Run select_best_tst_version() "
                "after compute_scores().")
        if self.target_styles:
            res = {}
            for fs_key in TARGET_STYLES:
                f_styles = TARGET_STYLES[fs_key]
                if set(f_styles) <= set(self.target_styles):
                    res[fs_key] = self.best_tst_results[
                        self.best_tst_results.target_style.isin(f_styles)
                    ][score_col].mean()
        else:
            res = {
                'DATA_MEAN': self.best_tst_results[score_col].mean()
            }
        # The run's evaluation encoder (task 5.2): scores of different
        # encoders are not comparable.
        res['evaluation_encoder'] = self._evaluation_encoder
        return res

    def compute_quality_scores(self, alignment_scorer, gender_scorer, entity_scorer) -> None:
        """Add bi_score, cl_score, bi_cl, gender_score, entity_score to self.tst_results.

        Operates on all generated versions (tst_results) so that score_v2 can be
        used with select_best_tst_version before any selection is applied.

        Parameters
        ----------
        alignment_scorer : AlignmentScorer
        gender_scorer    : GenderConsistencyScorer
        entity_scorer    : EntityConsistencyScorer
        """
        if self.tst_results is None:
            raise ValueError("tst_results is missing. Run produce_tst_results() first.")

        df = self.tst_results
        al = alignment_scorer.score_batch(df.text.tolist(), df.styled_text.tolist(),
                                          return_masks=False)
        df['bi_score']     = al['bi_score']
        df['cl_score']     = al['cl_score']
        df['bi_cl']        = 1.0 - np.clip(np.maximum(al['bi_score'], al['cl_score']), 0.0, 1.0)

        gn = gender_scorer.score_batch(df.text.tolist(), df.styled_text.tolist())
        df['gender_score'] = gn['gender_score']
        # Keep the activation flag: gender filtering applies only to pairs where
        # the source has a gendered pronoun (activated=True). score_v2 ignores it,
        # but the short-text gen pipeline filters activated-only (task 1.6).
        df['gender_activated'] = gn['activated']

        en = entity_scorer.score_batch(df.text.tolist(), df.styled_text.tolist())
        df['entity_score'] = en['entity_score']

        # Defragment the activations the (resident) scorers just produced, so a
        # long streaming run on a small GPU doesn't accumulate cache fragmentation.
        scoring._free_gpu()

    def compute_composite_v2(self) -> None:
        """Add meaning_nolen, nat_v2, score_v2 to self.tst_results.

        Required call order:
            metrics.compute_scores()             # style_score, bert/labse, perplexities
            metrics.compute_quality_scores(...)  # bi_cl, gender_score, entity_score
            metrics.compute_composite_v2()       # score_v2
        """
        if self.tst_results is None:
            raise ValueError("tst_results is missing. Run produce_tst_results() first.")

        df = self.tst_results
        missing_score = [c for c in _SCORE_COLS if c not in df.columns]
        if missing_score:
            raise ValueError(
                f"Missing columns {missing_score} — call compute_scores() first."
            )
        missing_quality = [c for c in _QUALITY_COLS if c not in df.columns]
        if missing_quality:
            raise ValueError(
                f"Missing columns {missing_quality} — call compute_quality_scores() first."
            )

        df['meaning_nolen'] = meaning_score(
            df['bert_score'], df['labse_score'], np.ones(len(df))
        )

        if 'target_style' in df.columns:
            target_styles = df['target_style']
        elif 'target_style_desc' in df.columns:
            target_styles = df['target_style_desc']
        else:
            warnings.warn(
                "Neither 'target_style' nor 'target_style_desc' found in tst_results — "
                "nat_v2 will use source_CE as reference for all rows (no author-CE lookup).",
                stacklevel=2,
            )
            target_styles = pd.Series([''] * len(df), index=df.index)
        df['nat_v2'] = compute_nat_v2(
            df['styled_text_perplexity'], df['text_perplexity'], target_styles
        )

        df['score_v2'] = (
            base_score_v2(
                df['style_score'], df['meaning_nolen'], df['bi_cl'],
                df['gender_score'], df['entity_score']
            ) * df['nat_v2']
        )

    def compute_copying_metrics(self, df=None):
        """Compute the chrF copying diagnostic on the target DataFrame
        (default: self.best_tst_results). Delegates to ``copying.add_chrf``;
        does NOT modify the composite score.
        """
        if df is None:
            if not hasattr(self, "best_tst_results") or self.best_tst_results is None:
                raise AttributeError(
                    "best_tst_results is missing. Run execute() first, "
                    "or pass a DataFrame explicitly."
                )
            df = self.best_tst_results

        return add_chrf(df)

    def execute(self, source_style_path=None) -> None:
        """Run the score_v1 pipeline, in order:

            add_source_ppx_and_emb()   # cache source perplexity/style/labse on test_df
            produce_tst_results()      # dispatch text-target-style vs target-style-emb
            compute_scores()           # style/meaning/naturality + score
            select_best_tst_version()  # best row per (example, target_style) by 'score'

        score_v2 extends this *after* execute() (it needs external scorers, so it
        is not part of execute):

            compute_quality_scores(alignment, gender, entity)  # bi_cl/gender/entity
            compute_composite_v2()                             # meaning_nolen/nat_v2/score_v2
            select_best_tst_version(score_col='score_v2')

        The downstream methods guard this order explicitly (raising if a required
        column is absent).

        Args:
            source_style_path: optional labelled parquet side file of
                source style vectors (overrides the constructor's). When
                given, every entry checks it against the evaluation encoder
                and fills the instance's source store from it; otherwise
                the instance encodes each unseen source text once. A
                stored `text_style_emb` that differs from the store is
                dropped with a warning and encoded fresh; the instance's
                own column is a copy of the store and passes.
        """
        if source_style_path is not None:
            self.source_style_path = source_style_path
        entry = "TstPerformanceMetrics.execute"
        if not self._emb_mode():
            # execute ends in compute_scores, so the text path's scoring
            # source is required before any encode (the emb path's source
            # is required by the shared validation)
            self._require_scoring_targets(entry)
        self.test_df, _ = self._validate_frame(self.test_df, entry)
        self.add_source_ppx_and_emb()
        self.produce_tst_results()
        self.compute_scores()
        self.select_best_tst_version()
