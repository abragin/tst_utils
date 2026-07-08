"""Style-encoder disentanglement benchmark (Phase Style-v2, Step 1).

Scores a style encoder on the documented failure-mode axes and emits a
machine-readable scorecard (one row per ``(axis, slice, metric)``).

Design contract (pinned in the task file before implementation):

- **Injectable encoder.** ``score_encoder`` takes an embedding callable
  ``embed(list[str]) -> np.ndarray`` or a SentenceTransformer-like object
  (anything with ``.encode``). It never reads
  ``tst_utils.eval.model_names.STYLE_MODEL``; the caller states
  ``encoder_id`` explicitly and it is stamped into every scorecard row.
- **Assets immutable, scoring pure.** Scoring only loads frozen labeled
  assets, computes embeddings with the requested encoder, and emits metric
  rows. It never calls generation or LLM judges. Each asset file's content
  hash is stamped into the rows it produced (``asset_version``).
- **Populations pinned per axis** (see the per-axis scoring functions).
- **Bootstrap CIs everywhere**, fixed seed, ``n`` recorded per row.

Axes implemented here:

- ``a_formal`` — genuine-transfer vs same-register-paraphrase AUC inside the
  formal-source blind spot, from the labeled ``ub_benchmark`` asset.
- ``b1`` — paradetox toxic<->neutral attribute sanity check (the encoder was
  never claimed to fail here; a strong score does NOT clear
  register-flattening).
- ``b2`` — formality-gold register axis (pikabu-native), the
  register-relevant probe; clean slice excludes strategy-(d) rows.
- ``a_informal`` / ``c`` — placeholders until their assets are constructed
  (Milestones 2/3 of the task).
- ``d_geometry`` — geometry-coherence diagnostic (Step 1.A follow-on, not
  in ``DEFAULT_AXES``): per-slice Spearman rho between style_distance and
  chrF over the same frozen assets. Measurement-only; interpretation is
  gated on the emitted chrF dispersion and source-length rows.
"""

import datetime
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCORECARD_COLUMNS = [
    'axis', 'slice', 'metric', 'value', 'n', 'ci_low', 'ci_high',
    'label_source', 'asset_version', 'encoder_id', 'created_at',
    'seed', 'n_boot',
]

# Pinned label sources for axis (a-formal); see ub_benchmark/MANIFEST.md.
# Round 1: 4-judge plurality (anchor FP 0.025). Validation: sonnet_val
# (anchor FP 0.08); deepseek_val/mimo_val are kept in the loaded frame but
# not used for scoring (FP 0.33 / disqualified).
A_FORMAL_ROUND1_LABEL = 'majority4'
A_FORMAL_VALIDATION_LABEL = 'sonnet_val'

DEFAULT_SEED = 20260706
DEFAULT_N_BOOT = 2000


# ---------------------------------------------------------------------------
# Encoder adapter
# ---------------------------------------------------------------------------

def resolve_embed_fn(encoder):
    """Normalize an encoder argument into ``embed(list[str]) -> np.ndarray``.

    Accepts a plain callable, or any object with a SentenceTransformer-style
    ``.encode(texts)`` method. Raw (unnormalized) vectors are fine: all
    similarity math below is angular / scale-invariant, and axis projections
    normalize internally.
    """
    if hasattr(encoder, 'encode'):
        return lambda texts: np.asarray(encoder.encode(list(texts)))
    if callable(encoder):
        return lambda texts: np.asarray(encoder(list(texts)))
    raise TypeError(
        'encoder must be an embed(texts)->ndarray callable or expose '
        f'.encode(); got {type(encoder)!r}'
    )


def sentence_transformer_encoder(model_name, **st_kwargs):
    """Convenience adapter: model name -> embed fn (lazy import, GPU-safe)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, **st_kwargs)
    return lambda texts: np.asarray(
        model.encode(list(texts), show_progress_bar=False)
    )


# ---------------------------------------------------------------------------
# Metric math (numpy only; no model imports)
# ---------------------------------------------------------------------------

def rowwise_sim(u, v):
    """Vectorized ``metrics.style.sim_measure`` over row-aligned matrices."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    un = u / np.linalg.norm(u, axis=1, keepdims=True)
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    cos = np.clip(np.sum(un * vn, axis=1), -1.0, 1.0)
    sim = 1 - np.arccos(cos) / np.pi
    return (sim + 1) / 2


def rank_auc(pos, neg):
    """AUC = P(pos > neg) via average ranks (Mann-Whitney, tie-corrected)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind='mergesort')
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # average ranks over ties
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fp_adjusted_auc(auc, anchor_fp):
    """Mixture-correct an observed AUC for judge false positives.

    If the positive class is a mixture of true transfers (weight ``1-fp``)
    and mislabeled paraphrases (weight ``fp``, contributing ~0.5 vs the
    anchor negatives), then ``auc_obs = (1-fp)*auc_true + fp*0.5``. Solve for
    ``auc_true``; clip to [0, 1]. ``fp`` is the measured judge FP rate on
    the known-paraphrase anchors (the parent-UB discount rule).
    """
    if anchor_fp is None or np.isnan(anchor_fp) or anchor_fp >= 1.0:
        return np.nan
    return float(np.clip((auc - 0.5 * anchor_fp) / (1.0 - anchor_fp), 0.0, 1.0))


def bootstrap_auc_ci(pos, neg, *, seed, n_boot, alpha=0.05):
    """Percentile bootstrap CI for ``rank_auc``, resampling each class."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = rank_auc(
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        )
    return (
        float(np.nanquantile(stats, alpha / 2)),
        float(np.nanquantile(stats, 1 - alpha / 2)),
    )


def bootstrap_stat_ci(values, stat_fn, *, seed, n_boot, alpha=0.05):
    """Percentile bootstrap CI for ``stat_fn`` over resampled ``values``."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_fn(rng.choice(values, size=len(values), replace=True))
    return (
        float(np.nanquantile(stats, alpha / 2)),
        float(np.nanquantile(stats, 1 - alpha / 2)),
    )


def average_ranks(values):
    """1-based ranks with ties assigned their average rank (as in SciPy)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman_rho(x, y):
    """Spearman rank correlation = Pearson on average ranks (numpy only).

    Returns ``nan`` for ``n < 3`` or when either input has zero rank
    variance (e.g. a constant-chrF slice) — no divide warning is emitted.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f'length mismatch: {len(x)} vs {len(y)}')
    if len(x) < 3:
        return np.nan
    rx, ry = average_ranks(x), average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return np.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def bootstrap_corr_ci(x, y, *, seed, n_boot, alpha=0.05):
    """Percentile bootstrap CI for ``spearman_rho``, resampling PAIRED rows.

    ``bootstrap_stat_ci`` resamples a single 1-D array and cannot express a
    two-vector paired statistic; this resamples pair *indices* and
    recomputes rho on both aligned columns (degenerate draws -> nan,
    excluded by the nan-quantiles).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = spearman_rho(x[idx], y[idx])
    return (
        float(np.nanquantile(stats, alpha / 2)),
        float(np.nanquantile(stats, 1 - alpha / 2)),
    )


# ---------------------------------------------------------------------------
# Frozen-asset loading
# ---------------------------------------------------------------------------

def file_version(path):
    """Short content hash stamped into scorecard rows (asset provenance)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


@dataclass
class BenchmarkAssets:
    """Paths to the frozen benchmark assets (live in the main repo)."""
    ub_formal_dir: str = None
    paradetox_path: str = None
    formality_gold_path: str = None
    ub_informal_dir: str = None      # Milestone 2 (a-informal)
    author_pairs_dir: str = None     # Milestone 3 (c)

    @classmethod
    def default(cls, project_root):
        infra = os.path.join(project_root, '16 phase 2A infrastructure')
        phase17 = os.path.join(project_root, '17 phase style-v2')
        return cls(
            ub_formal_dir=os.path.join(infra, 'ub_benchmark'),
            # VAL split only — the benchmark (b1 + paradetox d_geometry) must
            # never see the train rows the Step-3 retrain consumes. Physical
            # split materialised 2026-07-08 from the file's disjoint `split`
            # column (train 6,702 / val 759); train file is
            # ru_paradetox_pairs_train.parquet (training-only, never loaded here).
            paradetox_path=os.path.join(
                project_root, 'data', 'sources_v2', 'pool',
                'ru_paradetox_pairs_val.parquet'),
            formality_gold_path=os.path.join(
                infra, 'results', 'formality_gold_v1.parquet'),
            ub_informal_dir=os.path.join(phase17, 'ub_benchmark_informal'),
            author_pairs_dir=os.path.join(phase17, 'author_pairs_benchmark'),
        )


def load_ub_formal(ub_dir):
    """Load the formal (a) asset: round-1 and validation labeled frames.

    Returns ``(round1, validation, versions)`` where the frames carry
    ``pair_id, source, bucket, text, styled_text`` plus label columns, and
    ``versions`` maps logical name -> content hash of the files used.
    """
    ana = pd.read_csv(os.path.join(ub_dir, 'ub_label_analysis.csv'))
    inp = pd.read_csv(os.path.join(ub_dir, 'ub_label_input.csv'))
    round1 = ana.merge(inp, on='pair_id', validate='one_to_one')

    vin = pd.read_csv(os.path.join(ub_dir, 'ub_validation_input.csv'))
    vmap = pd.read_csv(os.path.join(ub_dir, 'ub_validation_bucket_map.csv'))
    validation = vin.merge(vmap, on='pair_id', validate='one_to_one')
    for judge in ('sonnet_val', 'deepseek_val', 'mimo_val'):
        jf = pd.read_csv(os.path.join(ub_dir, f'llm_review_{judge}.csv'))
        validation = validation.merge(
            jf[['pair_id', 'label']].rename(columns={'label': judge}),
            on='pair_id', validate='one_to_one')
    validation['source'] = 'news'  # validation round was news-only

    versions = {
        'ub_round1': file_version(
            os.path.join(ub_dir, 'ub_label_analysis.csv')),
        'ub_validation': file_version(
            os.path.join(ub_dir, 'ub_validation_bucket_map.csv')),
    }
    return round1, validation, versions


def load_paradetox(path):
    df = pd.read_parquet(path, columns=['text', 'styled_text'])
    return df.reset_index(drop=True), file_version(path)


def load_ub_informal(ub_dir):
    """Load the frozen a-informal asset (`ub_benchmark_informal/`).

    ``informal_label_analysis.csv`` carries the out-of-band design (source,
    bucket) + per-judge labels + the pre-registered aggregation column
    ``majority_usable``; texts are joined from the blind input file.
    """
    ana = pd.read_csv(os.path.join(ub_dir, 'informal_label_analysis.csv'))
    inp = pd.read_csv(os.path.join(ub_dir, 'informal_label_input.csv'))
    df = ana.merge(inp, on='pair_id', validate='one_to_one')
    version = file_version(os.path.join(ub_dir, 'informal_label_analysis.csv'))
    return df, version


def load_author_pairs(pairs_dir):
    """Load the frozen axis-(c) asset (built by mine_pikabu_author_pairs.py).

    Returns ``(pairs, chunks, writers, versions)``: the pikabu pair table
    (``kind, text_a, text_b, author_id_a/b, post_id_a/b``), the eval-partition
    chunk table (``post_id, author_id, text``), and the writers reference
    pair table (``kind, author_a/b, text_a, text_b``).
    """
    p_path = os.path.join(pairs_dir, 'pikabu_author_pairs_v1.parquet')
    c_path = os.path.join(pairs_dir, 'pikabu_author_chunks_eval.parquet')
    w_path = os.path.join(pairs_dir, 'writers_reference_pairs_v1.parquet')
    versions = {'pairs': file_version(p_path), 'chunks': file_version(c_path),
                'writers': file_version(w_path)}
    return (pd.read_parquet(p_path), pd.read_parquet(c_path),
            pd.read_parquet(w_path), versions)


def load_formality_gold(path):
    df = pd.read_parquet(
        path, columns=['strategy', 'text', 'formal_text', 'gold_pass'])
    df = df[df['gold_pass']].rename(
        columns={'formal_text': 'styled_text'}).reset_index(drop=True)
    return df, file_version(path)


# ---------------------------------------------------------------------------
# Per-axis scoring
# ---------------------------------------------------------------------------

def _row(axis, slice_, metric, value, n, ci, label_source, asset_version,
         *, seed=None, n_boot=None):
    ci_low, ci_high = ci if ci is not None else (np.nan, np.nan)
    return {
        'axis': axis, 'slice': slice_, 'metric': metric,
        'value': float(value) if value is not None else np.nan,
        'n': int(n),
        'ci_low': ci_low, 'ci_high': ci_high,
        'label_source': label_source, 'asset_version': asset_version,
        'seed': seed, 'n_boot': n_boot,
    }


def _score_transfer_vs_anchor(df, *, axis, slice_, label_col, embed_fn,
                              asset_version, seed, n_boot,
                              positive_query, anchor_query):
    """Shared (a)-axis machinery, population pinned explicitly per call.

    Positives = rows matching ``positive_query`` AND judged 'transfer' by
    ``label_col``. Negatives = ALL rows matching ``anchor_query`` (the
    register-preserving paraphrase anchors, known-paraphrase by
    construction — judge labels are NOT applied to the negative class).
    Signal = style movement ``1 - sim(text, styled_text)``; AUC = P(genuine
    transfer moves more than a paraphrase anchor). ``anchor_fp`` = fraction
    of anchors the judge mislabels 'transfer'; ``auc_movement_fp_adj``
    mixture-corrects the AUC by it (see ``fp_adjusted_auc``).
    """
    pos_df = df.query(positive_query)
    pos_df = pos_df[pos_df[label_col] == 'transfer']
    neg_df = df.query(anchor_query)
    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError(
            f'{axis}/{slice_}: empty class (pos={len(pos_df)}, '
            f'neg={len(neg_df)})')

    texts = pd.concat([pos_df, neg_df])
    src_emb = embed_fn(texts['text'].tolist())
    sty_emb = embed_fn(texts['styled_text'].tolist())
    movement = 1 - rowwise_sim(src_emb, sty_emb)
    mov_pos, mov_neg = movement[:len(pos_df)], movement[len(pos_df):]

    auc = rank_auc(mov_pos, mov_neg)
    ci = bootstrap_auc_ci(mov_pos, mov_neg, seed=seed, n_boot=n_boot)
    anchor_fp = float((neg_df[label_col] == 'transfer').mean())
    n = len(pos_df) + len(neg_df)

    rows = [
        _row(axis, slice_, 'auc_movement', auc, n, ci, label_col,
             asset_version, seed=seed, n_boot=n_boot),
        _row(axis, slice_, 'auc_movement_fp_adj',
             fp_adjusted_auc(auc, anchor_fp), n,
             (fp_adjusted_auc(ci[0], anchor_fp),
              fp_adjusted_auc(ci[1], anchor_fp)),
             label_col, asset_version, seed=seed, n_boot=n_boot),
        _row(axis, slice_, 'anchor_fp', anchor_fp, len(neg_df), None,
             label_col, asset_version),
        _row(axis, slice_, 'n_pos', len(pos_df), len(pos_df), None,
             label_col, asset_version),
        _row(axis, slice_, 'n_neg', len(neg_df), len(neg_df), None,
             label_col, asset_version),
    ]
    return rows


def score_axis_a_formal(embed_fn, assets, *, seed, n_boot):
    """Axis (a-formal): news and wikipedia scored separately, NO aggregate
    (wiki labels are weak/ambiguous per the MANIFEST); plus the news-only
    validation round (0.92–0.97 window rejects, sonnet_val labels)."""
    round1, validation, versions = load_ub_formal(assets.ub_formal_dir)
    rows = []
    for source in ('news', 'wikipedia'):
        rows += _score_transfer_vs_anchor(
            round1[round1['source'] == source],
            axis='a_formal', slice_=source,
            label_col=A_FORMAL_ROUND1_LABEL, embed_fn=embed_fn,
            asset_version=versions['ub_round1'], seed=seed, n_boot=n_boot,
            positive_query="bucket == 'reject'",
            anchor_query="bucket == 'rt_paraphrase'")
    rows += _score_transfer_vs_anchor(
        validation,
        axis='a_formal', slice_='news_validation',
        label_col=A_FORMAL_VALIDATION_LABEL, embed_fn=embed_fn,
        asset_version=versions['ub_validation'], seed=seed, n_boot=n_boot,
        positive_query="bucket == 'window_reject'",
        anchor_query="bucket == 'rt_paraphrase'")
    return rows


def _score_paired_axis(df, *, axis, slice_, label_source, embed_fn,
                       asset_version, seed, n_boot):
    """Shared (b)-axis machinery: attribute-direction probe on paired data.

    Population = 1-to-1 rewrite pairs sharing content (``text`` ->
    ``styled_text``); content is matched within every pair by construction.
    Pairs are split 50/50 (seeded): the attribute axis = mean normalized
    ``styled - text`` difference on the fit half; metrics are computed on
    the held-out half only (the axis stays fixed across bootstrap draws):

    - ``axis_auc``: pooled AUC of the axis projection separating styled from
      source texts.
    - ``paired_axis_acc``: fraction of held-out pairs where the styled side
      projects higher on the axis.
    - ``paired_step_median``: median within-pair style movement
      ``1 - sim(text, styled)``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    fit_idx, eval_idx = idx[:len(df) // 2], idx[len(df) // 2:]

    src = np.asarray(embed_fn(df['text'].tolist()), dtype=np.float64)
    sty = np.asarray(embed_fn(df['styled_text'].tolist()), dtype=np.float64)
    src_n = src / np.linalg.norm(src, axis=1, keepdims=True)
    sty_n = sty / np.linalg.norm(sty, axis=1, keepdims=True)

    axis_vec = (sty_n[fit_idx] - src_n[fit_idx]).mean(axis=0)
    axis_vec = axis_vec / np.linalg.norm(axis_vec)

    proj_src = src_n[eval_idx] @ axis_vec
    proj_sty = sty_n[eval_idx] @ axis_vec
    step = 1 - rowwise_sim(src[eval_idx], sty[eval_idx])

    auc = rank_auc(proj_sty, proj_src)
    auc_ci = bootstrap_auc_ci(proj_sty, proj_src, seed=seed, n_boot=n_boot)
    acc_vals = (proj_sty > proj_src).astype(float)
    acc_ci = bootstrap_stat_ci(acc_vals, np.mean, seed=seed, n_boot=n_boot)
    step_ci = bootstrap_stat_ci(step, np.median, seed=seed, n_boot=n_boot)
    n_eval = len(eval_idx)

    return [
        _row(axis, slice_, 'axis_auc', auc, 2 * n_eval, auc_ci,
             label_source, asset_version, seed=seed, n_boot=n_boot),
        _row(axis, slice_, 'paired_axis_acc', acc_vals.mean(), n_eval,
             acc_ci, label_source, asset_version, seed=seed, n_boot=n_boot),
        _row(axis, slice_, 'paired_step_median', float(np.median(step)),
             n_eval, step_ci, label_source, asset_version,
             seed=seed, n_boot=n_boot),
    ]


def score_axis_b1(embed_fn, assets, *, seed, n_boot, max_pairs=None):
    """Axis (b1): paradetox toxic<->neutral ATTRIBUTE SANITY CHECK.

    The encoder was never claimed to fail here; a strong score does NOT
    clear register-flattening (that evidence rests on b2 + the informal
    a/c axes). Reported separately from (b2), never fused."""
    df, version = load_paradetox(assets.paradetox_path)
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=seed).reset_index(drop=True)
    return _score_paired_axis(
        df, axis='b1_paradetox', slice_='all',
        label_source='paradetox_dataset', embed_fn=embed_fn,
        asset_version=version, seed=seed, n_boot=n_boot)


def score_axis_b2(embed_fn, assets, *, seed, n_boot):
    """Axis (b2): formality-gold register axis (informal->formal, pikabu).

    ``clean`` slice = strategy b_axis_informal only; ``strategy_d`` =
    d_near_paradetox_neutral rows, reported separately because of the
    documented contamination caveat (corpus-vs-register confound)."""
    df, version = load_formality_gold(assets.formality_gold_path)
    rows = []
    for slice_, strategy in (('clean', 'b_axis_informal'),
                             ('strategy_d', 'd_near_paradetox_neutral')):
        sub = df[df['strategy'] == strategy].reset_index(drop=True)
        rows += _score_paired_axis(
            sub, axis='b2_formality', slice_=slice_,
            label_source='formality_gold_v1(arbiter)', embed_fn=embed_fn,
            asset_version=version, seed=seed, n_boot=n_boot)
    return rows


A_INFORMAL_LABEL = 'majority_usable'  # pre-registered aggregation column


def score_axis_a_informal(embed_fn, assets, *, seed, n_boot):
    """Axis (a-informal): ficbook / pikabu / taiga_proza, per-source, no
    aggregate. Population = reject-bucket rows the pre-registered usable-judge
    majority labels 'transfer' vs ALL llm_paraphrase anchors (deepseek
    register-preserving paraphrases, pilot-validated)."""
    df, version = load_ub_informal(assets.ub_informal_dir)
    rows = []
    for source in ('ficbook', 'pikabu', 'taiga_proza'):
        rows += _score_transfer_vs_anchor(
            df[df['source'] == source],
            axis='a_informal', slice_=source,
            label_col=A_INFORMAL_LABEL, embed_fn=embed_fn,
            asset_version=version, seed=seed, n_boot=n_boot,
            positive_query="bucket == 'reject'",
            anchor_query="bucket == 'llm_paraphrase'")
    return rows


def _pair_sims(pairs, embed_fn):
    """Cosine-scale sim per pair row, embedding each unique text once."""
    uniq = pd.unique(pd.concat([pairs['text_a'], pairs['text_b']]))
    emb = np.asarray(embed_fn(list(uniq)), dtype=np.float64)
    idx = {t: i for i, t in enumerate(uniq)}
    a = emb[[idx[t] for t in pairs['text_a']]]
    b = emb[[idx[t] for t in pairs['text_b']]]
    return rowwise_sim(a, b)


def silhouette_by_group(emb, groups):
    """Per-sample silhouette on angular (sim_measure) distance, no sklearn.

    Groups with a single sample are excluded. Returns (values, group_labels)
    aligned arrays for the retained samples.
    """
    groups = np.asarray(groups)
    keep_groups = pd.Series(groups).value_counts()
    keep_groups = set(keep_groups[keep_groups >= 2].index)
    mask = np.array([g in keep_groups for g in groups])
    emb = np.asarray(emb, dtype=np.float64)[mask]
    groups = groups[mask]
    n = len(emb)
    un = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = np.clip(un @ un.T, -1.0, 1.0)
    dist = np.arccos(cos) / np.pi  # angular distance, matches sim_measure
    labels = pd.unique(groups)
    sil = np.empty(n)
    for i in range(n):
        own = groups == groups[i]
        a = dist[i, own].sum() / max(own.sum() - 1, 1)
        b = min(dist[i, groups == g].mean() for g in labels if g != groups[i])
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return sil, groups


def _bootstrap_grouped_mean_ci(values, groups, *, seed, n_boot, alpha=0.05):
    """CI for a grouped mean: resample groups, average their samples."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    by_group = {g: values[groups == g] for g in uniq}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        stats[b] = np.concatenate([by_group[g] for g in picked]).mean()
    return (float(np.nanquantile(stats, alpha / 2)),
            float(np.nanquantile(stats, 1 - alpha / 2)))


def score_axis_c(embed_fn, assets, *, seed, n_boot, max_sil_chunks=2000):
    """Axis (c): author identity in the OOD pikabu domain.

    Populations (pinned): primary AUC = ``same_post_same_author`` vs
    ``same_post_diff_author`` pair similarities (topic fixed in BOTH
    classes); secondary = ``cross_post_same_author`` vs
    ``same_post_diff_author`` (author signal must survive topic change).
    Silhouette = per-sample angular silhouette by ``author_id`` over the
    eval-partition chunks (authors with >=2 chunks; grouped bootstrap by
    author). The ``writers_reference`` slice recomputes the writers
    same/diff-author AUC with the SAME encoder — candidate encoders are
    compared against it, never against a historical number.

    Documented confounds (MANIFEST): thread-tone convergence deflates the
    same-post AUC and short comments add noise, so the relative base->v2
    delta is the trustworthy quantity, not the absolute level.
    """
    pairs, chunks, writers, versions = load_author_pairs(
        assets.author_pairs_dir)
    rows = []

    sims = _pair_sims(pairs, embed_fn)
    by_kind = {k: sims[(pairs['kind'] == k).values]
               for k in pairs['kind'].unique()}
    neg = by_kind['same_post_diff_author']
    for slice_, pos_kind in (
            ('pikabu_same_post', 'same_post_same_author'),
            ('pikabu_cross_post', 'cross_post_same_author')):
        pos = by_kind[pos_kind]
        auc = rank_auc(pos, neg)
        ci = bootstrap_auc_ci(pos, neg, seed=seed, n_boot=n_boot)
        n = len(pos) + len(neg)
        rows.append(_row('c_author', slice_, 'auc_same_author', auc, n, ci,
                         'pikabu_metadata', versions['pairs'],
                         seed=seed, n_boot=n_boot))

    wsims = _pair_sims(writers, embed_fn)
    wpos = wsims[(writers['kind'] == 'writers_same_author').values]
    wneg = wsims[(writers['kind'] == 'writers_diff_author').values]
    auc = rank_auc(wpos, wneg)
    ci = bootstrap_auc_ci(wpos, wneg, seed=seed, n_boot=n_boot)
    rows.append(_row('c_author', 'writers_reference', 'auc_same_author',
                     auc, len(wpos) + len(wneg), ci, 'writers_metadata',
                     versions['writers'], seed=seed, n_boot=n_boot))

    # silhouette over a seeded, author-balanced chunk sample
    counts = chunks['author_id'].value_counts()
    multi = chunks[chunks['author_id'].isin(counts[counts >= 2].index)]
    if len(multi) > max_sil_chunks:
        rng = np.random.default_rng(seed)
        keep_authors = []
        total = 0
        for aid in rng.permutation(multi['author_id'].unique()):
            keep_authors.append(aid)
            total += counts[aid]
            if total >= max_sil_chunks:
                break
        multi = multi[multi['author_id'].isin(keep_authors)]
    emb = np.asarray(embed_fn(multi['text'].tolist()))
    sil, sil_groups = silhouette_by_group(emb, multi['author_id'].values)
    ci = _bootstrap_grouped_mean_ci(sil, sil_groups, seed=seed,
                                    n_boot=n_boot)
    rows.append(_row('c_author', 'pikabu_silhouette', 'silhouette_author',
                     float(sil.mean()), len(sil), ci, 'pikabu_metadata',
                     versions['chunks'], seed=seed, n_boot=n_boot))
    return rows


# Pinned d_geometry bucket membership (task 2026-07-06): genuine transfers
# + accepted + the paraphrase anchors. ``near_copy`` is EXCLUDED — near-
# identical text gives chrf~1 and sim~1 trivially, manufacturing negative
# rho that says nothing about interior coherence. The ub_formal validation
# frame (0.92-0.97 window population, news-only) is excluded as well.
D_GEOMETRY_FORMAL_BUCKETS = ('reject', 'accepted', 'rt_paraphrase')
D_GEOMETRY_INFORMAL_BUCKETS = ('reject', 'accepted', 'llm_paraphrase')


def d_geometry_slices(assets):
    """Pinned per-slice pair populations for the d_geometry axis.

    Returns a list of dicts (``slice``, ``df`` with aligned
    ``pair_id/bucket/text/styled_text`` columns, ``source_asset``,
    ``asset_version``, ``label_source``). Slices are reported per-asset
    only — a naive pooled ``all`` would be dominated by between-domain
    location shifts in chrF and sim, so none is emitted.
    """
    round1, _validation, f_versions = load_ub_formal(assets.ub_formal_dir)
    informal, i_version = load_ub_informal(assets.ub_informal_dir)
    paradetox, p_version = load_paradetox(assets.paradetox_path)
    gold, g_version = load_formality_gold(assets.formality_gold_path)

    def norm(df, *, pair_id=None, bucket=None):
        out = pd.DataFrame({
            'pair_id': (df[pair_id].astype('float64').values
                        if pair_id else np.full(len(df), np.nan)),
            'bucket': df[bucket].astype(str).values if bucket else None,
            'text': df['text'].values,
            'styled_text': df['styled_text'].values,
        })
        return out

    slices = []
    for source in ('news', 'wikipedia'):
        sub = round1[(round1['source'] == source)
                     & round1['bucket'].isin(D_GEOMETRY_FORMAL_BUCKETS)]
        slices.append(dict(
            slice=source, df=norm(sub, pair_id='pair_id', bucket='bucket'),
            source_asset='ub_benchmark',
            asset_version=f_versions['ub_round1'],
            label_source='bucket_design'))
    for source in ('ficbook', 'pikabu', 'taiga_proza'):
        sub = informal[(informal['source'] == source)
                       & informal['bucket'].isin(D_GEOMETRY_INFORMAL_BUCKETS)]
        slices.append(dict(
            slice=source, df=norm(sub, pair_id='pair_id', bucket='bucket'),
            source_asset='ub_benchmark_informal', asset_version=i_version,
            label_source='bucket_design'))
    slices.append(dict(
        slice='formality', df=norm(gold, bucket='strategy'),
        source_asset='formality_gold_v1', asset_version=g_version,
        label_source='formality_gold_v1(arbiter)'))
    slices.append(dict(
        slice='paradetox', df=norm(paradetox),
        source_asset='ru_paradetox_pairs', asset_version=p_version,
        label_source='paradetox_dataset'))
    return slices


def score_axis_d_geometry(embed_fn, assets, *, seed, n_boot,
                          points_out=None, points_encoder_id=None):
    """Axis (d): geometry-coherence diagnostic — Spearman rho between
    ``style_distance`` (``1 - sim``) and chrF, per slice.

    For a coherent embedding space rho is NEGATIVE (minor rewording ->
    high chrF -> low style_distance). MEASUREMENT-ONLY: the directional
    guard lives in the notebook interpretation, gated on the emitted
    ``chrf_iqr`` (near-zero rho on a narrow-chrF slice is a power
    artifact, not a verdict) and on ``src_len_median_words`` (chrF swings
    wider per edit on short texts — the paradetox length confound). A
    chrF-prediction training head is explicitly rejected (see the task):
    this axis is never a loss.

    Populations pinned in ``d_geometry_slices``. All paradetox pairs are
    embedded (no subsample); each unique text is embedded once across all
    slices. Slices with ``n < 3`` emit nan (``n`` still recorded); an
    empty frozen slice raises. If ``points_out`` is given, a row-level
    ``(slice, source_asset, pair_id, bucket, chrf, style_distance,
    src_len_words, src_len_chars, encoder_id)`` parquet is written for
    notebook scatters, outlier audits, and the length-control check
    (length-matched / length-stratified rho re-estimates need per-row
    source lengths).
    """
    from tst_utils.eval.metrics.copying import chrf_scores  # lazy: sacrebleu

    slices = d_geometry_slices(assets)
    all_texts = pd.unique(pd.concat(
        [s['df'][col] for s in slices for col in ('text', 'styled_text')],
        ignore_index=True))
    emb = np.asarray(embed_fn(list(all_texts)), dtype=np.float64)
    tidx = {t: i for i, t in enumerate(all_texts)}

    rows, point_frames = [], []
    for s in slices:
        df = s['df']
        if len(df) == 0:
            raise ValueError(
                f"d_geometry/{s['slice']}: frozen slice came back empty")
        src = emb[[tidx[t] for t in df['text']]]
        sty = emb[[tidx[t] for t in df['styled_text']]]
        distance = 1 - rowwise_sim(src, sty)
        chrf = chrf_scores(df['text'].tolist(), df['styled_text'].tolist())
        n = len(df)
        if n >= 3:
            rho = spearman_rho(chrf, distance)
            ci = bootstrap_corr_ci(chrf, distance, seed=seed, n_boot=n_boot)
        else:
            rho, ci = np.nan, None
        q1, q3 = np.percentile(chrf, [25, 75])
        words = np.array([len(str(t).split()) for t in df['text']])
        chars = np.array([len(str(t)) for t in df['text']])
        meta = dict(label_source=s['label_source'],
                    asset_version=s['asset_version'])
        rows += [
            _row('d_geometry', s['slice'], 'spearman_rho', rho, n, ci,
                 seed=seed, n_boot=n_boot, **meta),
            _row('d_geometry', s['slice'], 'chrf_iqr', float(q3 - q1), n,
                 None, **meta),
            _row('d_geometry', s['slice'], 'src_len_median_words',
                 float(np.median(words)), n, None, **meta),
            _row('d_geometry', s['slice'], 'src_len_median_chars',
                 float(np.median(chars)), n, None, **meta),
        ]
        if points_out is not None:
            point_frames.append(pd.DataFrame({
                'slice': s['slice'], 'source_asset': s['source_asset'],
                'pair_id': df['pair_id'].values, 'bucket': df['bucket'],
                'chrf': chrf, 'style_distance': distance,
                'src_len_words': words, 'src_len_chars': chars,
                'encoder_id': points_encoder_id}))
    if points_out is not None:
        pd.concat(point_frames, ignore_index=True).to_parquet(points_out)
    return rows


AXIS_SCORERS = {
    'a_formal': score_axis_a_formal,
    'b1': score_axis_b1,
    'b2': score_axis_b2,
    'a_informal': score_axis_a_informal,
    'c': score_axis_c,
    'd_geometry': score_axis_d_geometry,
}

DEFAULT_AXES = ('a_formal', 'b1', 'b2')


# ---------------------------------------------------------------------------
# Entry point + scorecard writer
# ---------------------------------------------------------------------------

def score_encoder(encoder, *, encoder_id, assets, axes=DEFAULT_AXES,
                  seed=DEFAULT_SEED, n_boot=DEFAULT_N_BOOT, **axis_kwargs):
    """Score ``encoder`` on the benchmark axes; return scorecard DataFrame.

    ``encoder``: embed callable or SentenceTransformer-like (see
    ``resolve_embed_fn``). ``encoder_id``: caller-stated identifier stamped
    into every row (e.g. ``'abragin/ruBert-style-base'``). ``assets``:
    ``BenchmarkAssets`` (use ``BenchmarkAssets.default(project_root)``).
    ``axis_kwargs`` are forwarded per-axis (e.g. ``max_pairs`` for b1).
    """
    embed_fn = resolve_embed_fn(encoder)
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds')
    rows = []
    for axis in axes:
        if axis not in AXIS_SCORERS:
            raise KeyError(
                f'unknown axis {axis!r}; known: {sorted(AXIS_SCORERS)}')
        kwargs = dict(seed=seed, n_boot=n_boot)
        if axis == 'b1' and 'max_pairs' in axis_kwargs:
            kwargs['max_pairs'] = axis_kwargs['max_pairs']
        if axis == 'c' and 'max_sil_chunks' in axis_kwargs:
            kwargs['max_sil_chunks'] = axis_kwargs['max_sil_chunks']
        if axis == 'd_geometry' and 'd_geometry_points_out' in axis_kwargs:
            kwargs['points_out'] = axis_kwargs['d_geometry_points_out']
            kwargs['points_encoder_id'] = encoder_id
        rows += AXIS_SCORERS[axis](embed_fn, assets, **kwargs)
    df = pd.DataFrame(rows)
    df['encoder_id'] = encoder_id
    df['created_at'] = created_at
    return df[SCORECARD_COLUMNS]


def write_scorecard(df, out_prefix):
    """Write the canonical scorecard as ``<prefix>.json`` + ``<prefix>.csv``.

    The JSON (records orient) is the machine-readable source of truth the
    baseline notebook reads; the CSV is for eyeballing/diffing."""
    df = df[SCORECARD_COLUMNS]
    csv_path, json_path = out_prefix + '.csv', out_prefix + '.json'
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False,
                  indent=1, default=str)
    return json_path, csv_path


def read_scorecard(path):
    """Read a scorecard written by ``write_scorecard`` (json or csv)."""
    if path.endswith('.json'):
        with open(path) as f:
            return pd.DataFrame(json.load(f))[SCORECARD_COLUMNS]
    return pd.read_csv(path)[SCORECARD_COLUMNS]
