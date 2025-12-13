import numpy as np
from numpy.random import default_rng
import pandas as pd
import os
import torch
from transformers import AutoTokenizer
from datetime import datetime
from typing import Optional, List

from tst_utils.eval.performance import TstPerformanceMetrics
from tst_utils.eval.metrics.style import sim_measure
from tst_utils.tinystyler import TinyStyler
from tst_utils.tst_generator import TSTGenerator, GEN_OPTS_V3

SIM_MEASURE_UB = 0.92
SIM_MEASURE_TARGET_STYLE_UB = 0.9
NATURALITY_SCORE_LB = 0.8
MEANING_SCORE_LB = 0.75

# ---------------- Helper functions ----------------

def _as_array(vec):
    """Ensure embedding is a 1D numpy array."""
    return np.asarray(vec, dtype=float)

def to_array(x):
    """Convert Series or list of arrays to a 2D numpy array."""
    if isinstance(x, pd.Series):
        return np.stack(x.to_list())
    return np.stack(x)

def _col_to_array_list(series: pd.Series) -> List[np.ndarray]:
    """Convert column with embeddings (lists/arrays) to list of numpy arrays."""
    return [_as_array(x) for x in series.to_list()]

def _rescale_targets_to_source_norms(source_embs, target_embs):
    """Rescale each target embedding to have same norm as corresponding source embedding."""
    src_norms = np.array([np.linalg.norm(s) or 1.0 for s in source_embs])
    tgt_norms = np.linalg.norm(target_embs, axis=1)
    tgt_norms_safe = np.where(tgt_norms == 0, 1.0, tgt_norms)
    scaled = target_embs * (src_norms / tgt_norms_safe)[:, None]
    return [scaled[i] for i in range(scaled.shape[0])]

# ---------------- Main generic function ----------------

def produce_target_style(
    short_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    key_col: str,
    n_picks: int = 1,
    source_emb_col: str = "text_style_emb",
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    For each row in short_df:
      - choose n_picks random *different* values of key_col from complete_df, excluding the row's own value
      - sample one embedding per chosen group
      - mix them using random weights (if n_picks > 1)
      - rescale to norm of source embedding
    Returns: list of numpy arrays
    """
    if rng is None:
        rng = default_rng()
    # Pre-group pool embeddings for each key value
    grouped = {
        k: np.stack([_as_array(e) for e in g[source_emb_col].to_list()], axis=0)
        for k, g in complete_df.groupby(key_col)
    }
    all_keys = list(grouped.keys())

    # Extract input data
    src_embs = _col_to_array_list(short_df[source_emb_col])
    src_keys = short_df[key_col].to_list()

    target_embs = []

    for key, src_emb in zip(src_keys, src_embs):
        # Choose candidates excluding current key if possible
        candidates = [k for k in all_keys if k != key]
        if not candidates:
            candidates = all_keys

        # Pick n_picks distinct or with replacement if not enough
        replace = len(candidates) < n_picks
        picks = rng.choice(candidates, size=n_picks, replace=replace)

        # Sample one embedding from each group
        embs = [grouped[pick][rng.integers(0, len(grouped[pick]))] for pick in picks]

        # Weighted combination if n_picks > 1
        if n_picks == 1:
            combined = embs[0]
        else:
            weights = rng.random(n_picks)
            weights /= weights.sum()
            combined = np.sum([w * e for w, e in zip(weights, embs)], axis=0)

        target_embs.append(combined)

    target_embs = np.stack(target_embs)
    return _rescale_targets_to_source_norms(src_embs, target_embs)

# ---------------- Thin wrappers ----------------

def produce_target_style_random_writer(
    short_df, writers_df, author_col="author", source_emb_col="text_style_emb", rng=None
):
    return produce_target_style(
        short_df, writers_df, key_col=author_col, n_picks=1,
        source_emb_col=source_emb_col, rng=rng
    )

def produce_target_style_2_random_writers(
    short_df, writers_df, author_col="author", source_emb_col="text_style_emb", rng=None
):
    return produce_target_style(
        short_df, writers_df, key_col=author_col, n_picks=2,
        source_emb_col=source_emb_col, rng=rng
    )

def produce_target_style_other_domain(
    short_df, complete_df, domain_col="domain", source_emb_col="text_style_emb", rng=None
):
    return produce_target_style(
        short_df, complete_df, key_col=domain_col, n_picks=1,
        source_emb_col=source_emb_col, rng=rng
    )

def produce_target_style_2_other_domains(
    short_df, complete_df, domain_col="domain", source_emb_col="text_style_emb", rng=None
):
    return produce_target_style(
        short_df, complete_df, key_col=domain_col, n_picks=2,
        source_emb_col=source_emb_col, rng=rng
    )

# ---------- Perturbation-based style sampling ----------
def add_noise_to_embs(style_embs, scale=0.01, rng=None):
    """
    Add Gaussian noise to normalized embeddings to simulate style shifts.
    """
    if rng is None:
        rng = default_rng()
    style_embs = to_array(style_embs)

    norms = np.linalg.norm(style_embs, axis=1, keepdims=True)
    norm_embs = style_embs / norms
    noise = rng.normal(0, scale, norm_embs.shape)

    target_embs = norm_embs + noise
    target_embs = target_embs / np.linalg.norm(target_embs, axis=1, keepdims=True) * norms
    return list(target_embs)

def negate_some_vals(style_embs, quantile=0.02, rng=None):
    """
    Randomly negate a given proportion of values in embeddings.
    - quantile=0.02 → 2% of values negated on average.
    """
    if rng is None:
        rng = default_rng()
    style_embs = to_array(style_embs)

    mask = (rng.uniform(size=style_embs.shape) > quantile).astype(int) * 2 - 1
    target_embs = style_embs * mask
    return list(target_embs)

def sim_measure_series(df, target_emb_col, source_emb_col="text_style_emb"):
    return [
        sim_measure(se,te)
        for se,te in zip(df[source_emb_col], df[target_emb_col])
    ]
    
def get_target_styles(current_domain):
    base_domain_target_style_types = ['other_domain', '2_domains_weighted_avg']
    base_author_target_style_types = ['other_author', '2_authors_weighted_avg']
    if current_domain == 'news':
        base_target_styles = base_domain_target_style_types
    else:
        base_target_styles = (
            base_author_target_style_types + base_domain_target_style_types
        )
    target_styles = [
        bts + suffix
        for bts in base_target_styles
        for suffix in ["", "_with_negations", "_with_noise"]
    ]
    return target_styles

def add_target_style_emb(df, style_df, in_domain_style_df=None):
    for target_style_desc in df.target_style_desc.unique():
        df_target_style = df[df.target_style_desc == target_style_desc]
        if not df_target_style.empty:
            if 'other_author' in target_style_desc:
                target_style_embs = produce_target_style_random_writer(
                    df_target_style, in_domain_style_df
                )
            elif '2_authors_weighted_avg' in target_style_desc:
                target_style_embs = produce_target_style_2_random_writers(
                    df_target_style, in_domain_style_df
                )
            elif 'other_domain' in target_style_desc:
                target_style_embs = produce_target_style_other_domain(
                    df_target_style, style_df
                )
            elif '2_domains_weighted_avg' in target_style_desc:
                target_style_embs = produce_target_style_2_other_domains(
                    df_target_style, style_df
                )
            else:
                raise Exception('Unsupported target style type: ' + target_style_desc)
            if '_with_noise' in target_style_desc:
                target_style_embs = add_noise_to_embs(target_style_embs, scale=0.01)
            elif '_with_negations' in target_style_desc:
                qt = 0.02 if 'author' in target_style_desc else 0.01
                target_style_embs = negate_some_vals(target_style_embs, quantile=qt)
            df.loc[df_target_style.index, 'target_style_emb'] = pd.Series(
                target_style_embs, index=df_target_style.index, dtype=object
            )

def gen_paraphrases(current_df, tst_generator, style_df, in_domain_style_df, target_styles):
    current_df = current_df.copy()
    current_df['target_style_desc'] = np.random.choice(
        target_styles, size=current_df.shape[0]
    )
    add_target_style_emb(current_df, style_df, in_domain_style_df)
    current_df['target_style_sim_measure'] = sim_measure_series(
        current_df, 'target_style_emb'
    )
    current_df = current_df[
        current_df.target_style_sim_measure < SIM_MEASURE_TARGET_STYLE_UB
    ].copy()
    pm = TstPerformanceMetrics(
        test_df=current_df,
        tst_func=tst_generator.perform_tst,
        target_styles=None, #TARGET_STYLES['COMPLETE'],
        tst_model='pph_gen_with_random_target_style',
        author_styles=None, #author_styles,
        verbose=True
    )
    pm.execute()
    tst_res = pm.best_tst_results.set_index(
        pm.best_tst_results.example_number
    )
    tst_res.index.name = None
    expected = set(current_df.index)
    actual = set(tst_res.index)
    extra = actual - expected
    if extra:
        raise ValueError(f"TST generator returned invalid indices: {extra}")
    tst_res['tst_result_style_sim'] = sim_measure_series(
        tst_res, 'styled_text_style_emb'
    )
    tst_res = tst_res[
        (tst_res.meaning_score > MEANING_SCORE_LB) &
        (tst_res.naturality_score > NATURALITY_SCORE_LB) &
        (tst_res.tst_result_style_sim < SIM_MEASURE_UB)
    ]
    return tst_res

def save_tst_results(tst_df, results_path):
    tst_res_cols = [
        'target_style_desc','styled_text',
        'styled_text_style_emb', 'meaning_score',
        'tst_result_style_sim'
    ]
    results_files = [
        fn
        for fn in os.listdir(results_path)
        if '.parquet.gzip' in fn
    ]
    if results_files:
        nums = [
            int(fn.split('.')[0].split('_')[1])
            for fn in results_files
        ]
        next_result_num = max(nums) + 1
    else:
        next_result_num = 1
    next_result_str_num = str(next_result_num).zfill(5)
    new_result_path = results_path + f"part_{next_result_str_num}.parquet.gzip"
    tst_df[tst_res_cols].to_parquet(new_result_path, compression='gzip')

def join_generated_files(results_path):
    results_files = [
        fn
        for fn in os.listdir(results_path)
        if '.parquet.gzip' in fn
    ]
    if len(results_files) > 0:
        results_df = pd.concat(
            [
                pd.read_parquet(results_path + fn)
                for fn in results_files
            ]
        ).sort_index()
        if len(results_files) > 1:
            new_file_path = results_path + "part_00001_new.parquet.gzip"
            results_df.to_parquet(new_file_path, compression='gzip')
            for fn in results_files:
                os.remove(results_path + fn)
            os.rename(new_file_path, results_path + "part_00001.parquet.gzip")
        return results_df
    return None

class PphGenerator:
    def __init__(
        self,
        style_df_path,
        base_df_path,
        results_path,
        trained_model_path,
        long_texts=False,
        rows_at_once=30000,
        base_model_path = "ai-forever/rugpt3small_based_on_gpt2"
    ):
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_df = pd.read_parquet(base_df_path)
        self.style_df = pd.read_parquet(style_df_path)
        self.results_path = results_path
        self.rows_at_once = rows_at_once
        current_domain = self.base_df.iloc[0].domain
        self.target_styles = get_target_styles(current_domain)
        if current_domain == 'news':
            self.in_domain_style_df = None
        else:
            self.in_domain_style_df = self.style_df[self.style_df.domain == current_domain]
        if long_texts:
            max_length = 1024
            batch_size = 6
        else:
            max_length = 256
            batch_size = 10
        model = TinyStyler(model_name=base_model_path, model_type='GPT', use_style=True).cuda()
        model.load_state_dict(torch.load(trained_model_path+'/pytorch_model.bin'))
        model.eval()        
        self.tst_generator = TSTGenerator(
            model,
            tokenizer,
            target_styles=None,
            model_type="GPT",  # "T5" or "GPT"
            style_emb_dict=None, #writers_style_embs,  # if None → use author tags
            batch_size=batch_size,
            num_sequences=1,
            generate_options=GEN_OPTS_V3,
            # max_length=512,
            max_length=max_length,
            # min_length=32,
        )

    def execute(self):
        generated_df = join_generated_files(self.results_path)
        processed_ids = generated_df.index if (generated_df is not None) else []
        unprocessed_queue = list(set(self.base_df.index) - set(processed_ids))
        np.random.shuffle(unprocessed_queue)
        while len(unprocessed_queue) > 99:
            time_start = datetime.now()
            processing_ids = unprocessed_queue[:self.rows_at_once]
            unprocessed_queue = unprocessed_queue[self.rows_at_once:]
            current_df = self.base_df.loc[processing_ids]
            tst_df = gen_paraphrases(
                current_df, self.tst_generator,
                self.style_df, self.in_domain_style_df,
                self.target_styles
            )
            save_tst_results(tst_df, self.results_path)
            completed_ids = set(tst_df.index)
            failed_ids = [
                i for i in processing_ids
                if i not in completed_ids
            ]
            unprocessed_queue += failed_ids
            time_end = datetime.now()
            time_str = time_end.strftime('%Y-%m-%d %H:%M:%S')
            n_tried = len(processing_ids)
            n_succeeded = len(completed_ids)
            n_unprocessed = len(unprocessed_queue)
            n_total = self.base_df.shape[0]
            n_processed = n_total - n_unprocessed
            success_p = f"{n_succeeded/n_tried:.1%}"
            print(
                f"{n_succeeded} from {n_tried} ({success_p}) generated paraphrases passed during last step."
            )
            print(f"Step time: {str(time_end-time_start)}")
            print(f"Totally: {time_str} processed {n_processed} from {n_total}")