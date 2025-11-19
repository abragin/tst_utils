import pandas as pd
import numpy as np
from tst_utils.eval.metrics.naturality import calculate_perplexity, naturality_score
from tst_utils.eval.metrics.meaning import (
    meaning_score, b_score, labse_scores_from_embs,
    calc_labse_embeddings, length_penalty
)
from tst_utils.eval.metrics.style import calc_style_embeddings, add_away_towards


TARGET_STYLES = {
    'DT': ['Dostoevsky', 'Tolstoy'],
    'BCDT': ['Bible', 'Chekhov', 'Dostoevsky', 'Tolstoy'],
    'COMPLETE': ['Bible', 'Chekhov', 'Dostoevsky', 'Tolstoy', 'News']
}

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
    - verbose (bool): Whether to print debug information during processing.
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

    def produce_tst_results_with_text_target_styles(self) -> None:
        res_dfs = []

        precalculated_cols = ["text_perplexity", "text_style_emb", "text_labse_emb"]
        cols_to_copy = (
            ['example_number', 'text', 'author'] +
            [c for c in precalculated_cols if c in self.test_df]
        )

        for t_style in self.target_styles:
            if self.verbose:
                print("Performing TST for style:", t_style)

            df = self.test_df[self.test_df.author != t_style].copy()
            df = df.reset_index(drop=False).rename(columns={"index": "example_number"})
            df["target_style"] = t_style
            df["tst_model"] = self.tst_model

            # Run TST
            tst_output = self.tst_func(df.text.tolist(), t_style)

            # --- Handle both 1D and 2D outputs ---
            if not tst_output:
                raise ValueError("tst_func returned an empty result.")

            # Detect output shape
            if isinstance(tst_output[0], list):  # 2D: [[ver1, ver2, ...], ...]
                num_versions = len(tst_output[0])

                # Sanity check: all have same length
                if not all(len(x) == num_versions for x in tst_output):
                    raise ValueError("Inconsistent number of generated versions per example.")

                flat_records = []
                for ex_idx, versions in enumerate(tst_output):
                    for v_idx, text in enumerate(versions):
                        flat_records.append(
                            df.loc[ex_idx, cols_to_copy].to_dict()
                            | {
                                "target_style": t_style,
                                "tst_model": self.tst_model,
                                "styled_text": text,
                                "version_number": v_idx,
                            }
                        )
                df_expanded = pd.DataFrame(flat_records)

            else:  # 1D: [output_text_1, output_text_2, ...]
                if len(tst_output) != len(df):
                    raise ValueError(
                        f"Length mismatch: tst_func returned {len(tst_output)} texts for {len(df)} examples."
                    )
                df["styled_text"] = tst_output
                df["version_number"] = 0
                df_expanded = df

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

        if self.verbose:
            print("Performing TST with target_style_embeddings")

        tst_output = self.tst_func(
            texts=df.text.tolist(),
            target_style_embeddings=df.target_style_emb.tolist()
        )

        if not tst_output:
            raise ValueError("tst_func returned an empty result.")

        if isinstance(tst_output[0], list):  # multiple generations per example
            num_versions = len(tst_output[0])
            if not all(len(x) == num_versions for x in tst_output):
                raise ValueError("Inconsistent number of generated versions per example.")

            flat_records = []
            for ex_idx, versions in enumerate(tst_output):
                for v_idx, text in enumerate(versions):
                    flat_records.append(
                        df.loc[ex_idx, cols_to_copy].to_dict()
                        | {
                            "tst_model": self.tst_model,
                            "styled_text": text,
                            "version_number": v_idx,
                        }
                    )
            self.tst_results = pd.DataFrame(flat_records)

        else:  # single output per example
            if len(tst_output) != len(df):
                raise ValueError(
                    f"Length mismatch: tst_func returned {len(tst_output)} texts for {len(df)} examples."
                )
            df["styled_text"] = tst_output
            df["version_number"] = 0
            self.tst_results = df

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
        if not ('text_perplexity' in self.test_df):
            self.test_df['text_perplexity'] = calculate_perplexity(self.test_df.text)[0]
        if not ('text_style_emb' in self.test_df):
            self.test_df['text_style_emb'] = calc_style_embeddings(
                self.test_df.text
            )
        if not ('text_labse_emb' in self.test_df):
            self.test_df['text_labse_emb'] = [
                e.cpu().numpy()
                for e in calc_labse_embeddings(self.test_df.text)
            ]

    def compute_scores(self):
        if self.tst_results is None:
            raise Exception("TST results are absent.")
        if self.verbose:
            print('Adding perplexity')
        if not ('text_perplexity' in self.tst_results):
            self.tst_results['text_perplexity'] = calculate_perplexity(self.tst_results.text)[0]
        self.tst_results['styled_text_perplexity'] = calculate_perplexity(self.tst_results.styled_text)[0]
        if self.verbose:
            print('Adding bert score')
        self.tst_results['bert_score'] = b_score(
            self.tst_results.text, self.tst_results.styled_text
        )
        if self.verbose:
            print('Adding labse score')
        if 'text_labse_emb' in self.tst_results:
            text_labse_embs = self.tst_results.text_labse_emb
        else:
            text_labse_embs = calc_labse_embeddings(self.tst_results.text)
        styled_text_labse_embs = calc_labse_embeddings(self.tst_results.styled_text)
        self.tst_results['labse_score'] = labse_scores_from_embs(
            text_labse_embs,
            styled_text_labse_embs
        )
        if self.verbose:
            print('Adding style embeddings')
        if not ('text_style_emb' in self.tst_results):
            self.tst_results['text_style_emb'] = calc_style_embeddings(
                self.tst_results.text
            )
        self.tst_results['styled_text_style_emb'] = calc_style_embeddings(
            self.tst_results.styled_text
        )
        add_away_towards(self.tst_results, self.author_styles)
        if self.verbose:
            print('Adding scores')
        self.tst_results['style_score'] = np.sqrt(
            self.tst_results.away * self.tst_results.towards
        )
        length_penalties = length_penalty(self.tst_results.text, self.tst_results.styled_text)
        self.tst_results['meaning_score'] = meaning_score(
            self.tst_results.bert_score,
            self.tst_results.labse_score,
            length_penalties
        )

        self.tst_results['naturality_score'] = naturality_score(
            self.tst_results.text_perplexity,
            self.tst_results.styled_text_perplexity
        )
        self.tst_results['score'] = (
            self.tst_results['style_score'] *
            self.tst_results['meaning_score'] *
            self.tst_results['naturality_score']
        ) ** (1./3)

    def select_best_tst_version(self) -> None:
        """
        For each (example_number, target_style) pair, select the row with the highest 'score'
        from self.tst_results and store the resulting DataFrame in self.best_tst_results.
        """

        if not hasattr(self, "tst_results") or self.tst_results is None:
            raise AttributeError("self.tst_results is missing. Run produce_tst_results() first.")

        if "score" not in self.tst_results.columns:
            raise ValueError("Column 'score' not found in self.tst_results. Run compute_scores() first.")

        # Ensure numeric score (in case it's stored as string)
        self.tst_results["score"] = pd.to_numeric(self.tst_results["score"], errors="coerce")

        # Drop rows with NaN scores to avoid errors
        valid_df = self.tst_results.dropna(subset=["score"])

        if valid_df.empty:
            raise ValueError("No valid 'score' values found in tst_results.")

        # Select row with the highest score for each (example_number and target_style (if present))
        if "target_style" in valid_df:
            idx = valid_df.groupby(["example_number", "target_style"])["score"].idxmax()
        else:
            idx = valid_df.groupby(["example_number"])["score"].idxmax()
        self.best_tst_results = valid_df.loc[idx].reset_index(drop=True)

    def final_performance(self):
        if self.target_styles:
            res = {}
            for fs_key in TARGET_STYLES:
                f_styles = TARGET_STYLES[fs_key]
                if set(f_styles) <= set(self.target_styles):
                    res[fs_key] = self.best_tst_results[
                        self.best_tst_results.target_style.isin(f_styles)
                    ].score.mean()
            return res
        else:
            return {
                'DATA_MEAN': self.best_tst_results.score.mean()
            }

    # Pefrorm TST, compute scores and select best version
    def execute(self) -> None:
        self.add_source_ppx_and_emb()
        self.produce_tst_results()
        self.compute_scores()
        self.select_best_tst_version()