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
from tst_utils.eval.performance.constants import TARGET_STYLES, _SCORE_COLS, _QUALITY_COLS
from tst_utils.eval.performance.reshape import expand_tst_output
from tst_utils.eval.performance.source_cache import ensure_source_caches
from tst_utils.eval.performance.copying import add_chrf
from tst_utils.eval.performance import scoring

logger = logging.getLogger(__name__)


class TstPerformanceMetrics:
    """
    Class to evaluate TST models using style transfer, meaning, and naturality metrics.

    Parameters:
    - test_df (pd.DataFrame): Dataset containing source texts and related metadata.
        Expected columns:
          • For TST with string target styles: 'text' and 'author'.
          • For TST with explicit target style embeddings:
              'text', 'author', 'target_style_emb', and 'target_style_desc'.
    - tst_func (Callable): Function performing text style transfer.
        • For TST with string target styles: should have a signature like (texts: List[str], target_style: str) -> List[str].
        • For TST with explicit target style embeddings: should have a signature like (texts: List[str], target_style_embeddings: List[np.ndarray]) -> List[str].
    - target_styles (List[str] | None): List of target styles to evaluate.
        Used only for text target style TST. If None, the class expects explicit target style embeddings in test_df.
    - tst_model (str): Name or ID of the TST model being evaluated.
    - author_styles (Dict[str, np.ndarray]): Precomputed style embeddings per author or domain. Used with target style TST only.
    - verbose (bool): Whether to log debug/progress information during processing.
    """
    def __init__(
        self, test_df, tst_func, target_styles, tst_model,
        author_styles, verbose = False
    ):
        self.test_df = test_df
        self.tst_func = tst_func
        self.target_styles = target_styles
        self.tst_model = tst_model
        self.verbose = verbose
        self.author_styles = author_styles
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

    def _log(self, msg, *args) -> None:
        """Verbose-gated progress logging (replaces the old `if self.verbose: print`)."""
        if self.verbose:
            logger.info(msg, *args)

    def produce_tst_results_with_text_target_styles(self) -> None:
        res_dfs = []

        precalculated_cols = ["text_perplexity", "text_style_emb", "text_labse_emb"]
        cols_to_copy = (
            ['example_number', 'text', 'author'] +
            [c for c in precalculated_cols if c in self.test_df]
        )

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
                extra_fields={"target_style": t_style, "tst_model": self.tst_model},
            )
            res_dfs.append(df_expanded)

        # Combine all target styles
        self.tst_results = pd.concat(res_dfs, ignore_index=True)

    def produce_tst_results_with_target_style_emb(self) -> None:
        """
        Perform TST using explicit style embeddings from test_df.
        Expects `target_style_emb` and `target_style_desc` columns in test_df.
        """
        if "target_style_emb" not in self.test_df or "target_style_desc" not in self.test_df:
            raise ValueError("test_df must contain 'target_style_emb' and 'target_style_desc' columns.")

        df = self.test_df.reset_index(drop=False).rename(columns={"index": "example_number"})
        df["tst_model"] = self.tst_model

        precalculated_cols = ["text_perplexity", "text_style_emb", "text_labse_emb"]
        cols_to_copy = (
            ['example_number', 'text', 'author', 'target_style_emb', 'target_style_desc'] +
            [c for c in precalculated_cols if c in self.test_df]
        )

        self._log("Performing TST with target_style_embeddings")

        tst_output = self.tst_func(
            texts=df.text.tolist(),
            target_style_embeddings=df.target_style_emb.tolist()
        )

        self.tst_results = expand_tst_output(
            tst_output, df, cols_to_copy,
            extra_fields={"tst_model": self.tst_model},
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
        elif {"target_style_emb", "target_style_desc"} <= set(self.test_df.columns):
            return self.produce_tst_results_with_target_style_emb()
        else:
            raise ValueError(
                "Cannot determine TST mode: either provide target_styles "
                "or ensure test_df has both 'target_style_emb' and 'target_style_desc' columns."
            )

    def add_source_ppx_and_emb(self):
        ensure_source_caches(
            self.test_df, perplexity=True, style_emb=True, labse_emb=True
        )

    def compute_scores(self):
        """Add base v1 scores to self.tst_results (style/meaning/naturality + score).

        Pipeline (each step caches/derives in place on self.tst_results):
            perplexity → bert → labse → style embeddings/away-towards → combine.
        """
        if self.tst_results is None:
            raise Exception("TST results are absent.")
        scoring.add_perplexity_scores(self.tst_results, self._log)
        scoring.add_bert_score(self.tst_results, self._log)
        scoring.add_labse_score(self.tst_results, self._log)
        scoring.add_style_scores(self.tst_results, self.author_styles, self._log)
        scoring.combine_scores(self.tst_results, self._log)

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
        if self.target_styles:
            res = {}
            for fs_key in TARGET_STYLES:
                f_styles = TARGET_STYLES[fs_key]
                if set(f_styles) <= set(self.target_styles):
                    res[fs_key] = self.best_tst_results[
                        self.best_tst_results.target_style.isin(f_styles)
                    ][score_col].mean()
            return res
        else:
            return {
                'DATA_MEAN': self.best_tst_results[score_col].mean()
            }

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

        en = entity_scorer.score_batch(df.text.tolist(), df.styled_text.tolist())
        df['entity_score'] = en['entity_score']

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

    def execute(self) -> None:
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
        """
        self.add_source_ppx_and_emb()
        self.produce_tst_results()
        self.compute_scores()
        self.select_best_tst_version()
