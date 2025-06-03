import pandas as pd
import numpy as np
from tst_utils.eval.metrics.naturality import calculate_perplexity, naturality_score
from tst_utils.eval.metrics.meaning import meaning_score, b_score, labse_meaning_score
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
    - test_df (pd.DataFrame): Dataset with 'text' and 'author'.
    - tst_func (Callable[[List[str], str], List[str]]): Function that takes texts and target style.
    - target_styles (List[str]): List of target styles for TST.
    - tst_model (str): Name or ID of the model being evaluated.
    - author_styles (Dict[str, np.ndarray]): Precomputed style embeddings.
    - verbose (bool): Whether to print debug info.
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

    def produce_tst_results(self) -> None:
        res_dfs = []
        for t_style in self.target_styles:
            if self.verbose:
                print("Performing TST for style: ", t_style)
            df = self.test_df[self.test_df.author != t_style].copy()
            df['target_style'] = t_style
            df['tst_model'] = self.tst_model
            df['styled_text'] = self.tst_func(df.text, t_style)
            res_dfs.append(df)
        self.tst_results = pd.concat(res_dfs, ignore_index=True)

    def compute_scores(self):
        if self.tst_results is None:
            raise Exception("TST results are absent.")
        if self.verbose:
            print('Adding perplexity')
        self.tst_results['source_perplexity'] = calculate_perplexity(self.tst_results.text)[0]
        self.tst_results['target_perplexity'] = calculate_perplexity(self.tst_results.styled_text)[0]
        if self.verbose:
            print('Adding bert score')
        self.tst_results['bert_score'] = b_score(
            self.tst_results.text, self.tst_results.styled_text
        )
        if self.verbose:
            print('Adding labse score')
        self.tst_results['labse_score'] = labse_meaning_score(self.tst_results.text, self.tst_results.styled_text)
        if self.verbose:
            print('Adding style embeddings')
        self.tst_results['text_emb'] = calc_style_embeddings(
            self.tst_results.text
        )
        self.tst_results['styled_text_emb'] = calc_style_embeddings(
            self.tst_results.styled_text
        )
        add_away_towards(self.tst_results, self.author_styles)
        if self.verbose:
            print('Adding scores')
        self.tst_results['style_score'] = np.sqrt(
            self.tst_results.away * self.tst_results.towards
        )
        self.tst_results['meaning_score'] = meaning_score(
            self.tst_results.bert_score,
            self.tst_results.labse_score
        )

        self.tst_results['naturality_score'] = naturality_score(
            self.tst_results.source_perplexity,
            self.tst_results.target_perplexity
        )
        self.tst_results['score'] = (
            self.tst_results['style_score'] *
            self.tst_results['meaning_score'] *
            self.tst_results['naturality_score']
        ) ** (1./3)

    def final_performance(self):
        res = {}
        for fs_key in TARGET_STYLES:
            f_styles = TARGET_STYLES[fs_key]
            if set(f_styles) <= set(self.target_styles):
                res[fs_key] = self.tst_results[
                    self.tst_results.target_style.isin(f_styles)
                ].score.mean()
        return res
