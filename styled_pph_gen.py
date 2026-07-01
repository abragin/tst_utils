import gc
import numpy as np
from numpy.random import default_rng
import pandas as pd
import os
import torch
from transformers import AutoTokenizer
from datetime import datetime
from typing import Optional, List

from tst_utils.eval.performance import TstPerformanceMetrics
from tst_utils.eval.performance.source_cache import ensure_source_caches
from tst_utils.eval.metrics.style import sim_measure, calc_style_embeddings
from tst_utils.eval.metrics.composite import compute_nat_v2
from tst_utils.eval.metrics.alignment import BatchedAligner
from tst_utils.eval.metrics.gender_consistency import GenderConsistencyScorer
from tst_utils.eval.metrics.entity_consistency import EntityConsistencyScorer
from tst_utils.tinystyler import TinyStyler
from tst_utils.tst_generator import TSTGenerator, GEN_OPTS_V3
from tst_utils.utils import set_global_seed

# Legacy v1 filters (retained).
SIM_MEASURE_UB = 0.92
SIM_MEASURE_TARGET_STYLE_UB = 0.9
MEANING_SCORE_LB = 0.75

# Phase-1 quality filters wired in 2A.4 (thresholds calibrated in task 1.10 on
# the mBERT-span4 regime; all are REJECTION conditions, so the keep-mask negates
# them). The absolute naturality_score is not used as a Phase-1 gate here — the
# relative nat_v2 (source-referenced CE gap) replaces it. (The earlier "rejects
# 88.5% of fluent literary transfers" figure was an artifact of the left-padding
# CE bug; naturality_score itself is correct on the fixed-CE scale and is retained
# only in the v1 score. See docs/issues/resolved/calculate-perplexity-left-pad-batch-dependence.md.)
CHRF_UB = 0.75          # keep chrf < 0.75 (copying diagnostic)
BI_SCORE_UB = 0.12      # keep bi_score < 0.12 (span4 boilerplate insertion)
CL_SCORE_UB = 0.10      # keep cl_score < 0.10 (span4 content loss)
GENDER_SCORE_LB = 0.70  # keep gender_score >= 0.70, activated pairs only
ENTITY_SCORE_LB = 0.70  # keep entity_score >= 0.70
# nat_v2 = 1/(max(0, style_gap - margin) + 1), margin=0.70 (composite.py). Soft
# generation gate: 0.75 rejects ~3% of best-of-1 generations (delta_CE > 1.03),
# firing only on the extreme low-fluency tail. Recalibrated 2026-07-01 from 0.5
# (which was inert post-CE-fix). See 15 metrics exploration/13 nat_v2 margin recalibration.ipynb.
NAT_V2_LB = 0.75        # keep nat_v2 >= 0.75 (source-referenced CE gap; soft)

# rugpt3 perplexity batch size for scoring. The per-batch logits tensor is the
# only thing that overflows tallin's ~8 GB GPU with the generator resident;
# 8 (peak ~1.6 GB) fits, 32 (peak ~6.6 GB) does not (decision 13).
PERPLEXITY_BATCH_SIZE = 8

DEFAULT_SEED = 1991

# Domain-only target domains: no per-author targets in the style pool. Only
# 'writers' retains author+domain targets; every other domain gets domain-only
# targets. 'political' carries author names (Putin/Medvedev) but is
# domain-targets-only by decision. ficbook/taiga_proza/pikabu were demoted here
# by the 2A.8.2 author-separability diagnostic: their per-author style centroids
# do not separate under the style encoder (silhouette −0.10/−0.19/−0.18 vs
# writers +0.15), and ficbook's deployed pool support is ~1 chunk/author, so an
# "author target" is operationally a domain draw. Derive has_author_targets from
# this explicit set, NOT from style-pool author cardinality: cardinality wrongly
# flips 'political' (2 authors) to author-targets and a real-author domain that
# happens to sample a single author to domain-only.
DOMAIN_ONLY = {'news', 'wikipedia', 'taiga_magazines', 'Bible', 'political',
               'ficbook', 'taiga_proza', 'pikabu'}

# ---- 2A.8.2 per-source target-domain weighting ----
# Domain-centroid cosine similarity (9x9, symmetric), measured on the final v2
# source pools (1000 chunks/domain, normalized-mean centroids, style encoder
# `abragin/ruBert-style-base`, seed 20260701; see notebook 16/14). Used to
# down-weight target domains that are too close to the source (near-identity,
# weak-transfer pairs) — e.g. news<->wikipedia 0.965, ficbook<->pikabu 0.943.
# The per-pair SIM_MEASURE_TARGET_STYLE_UB filter catches literal near-duplicate
# embeddings (drops 71% of news<->wiki) but is nearly blind to same-register
# non-duplicate pairs (16% of ficbook<->pikabu); this weighting covers that gap.
DOMAIN_CENTROID_SIM = {
    'news':            {'news': 1.0000, 'wikipedia': 0.9652, 'ficbook': 0.5094, 'pikabu': 0.6555, 'taiga_proza': 0.5734, 'taiga_magazines': 0.6505, 'writers': 0.5271, 'Bible': 0.4305, 'political': 0.5698},
    'wikipedia':       {'news': 0.9652, 'wikipedia': 1.0000, 'ficbook': 0.6739, 'pikabu': 0.7914, 'taiga_proza': 0.7523, 'taiga_magazines': 0.8169, 'writers': 0.6901, 'Bible': 0.5123, 'political': 0.6615},
    'ficbook':         {'news': 0.5094, 'wikipedia': 0.6739, 'ficbook': 1.0000, 'pikabu': 0.9428, 'taiga_proza': 0.9414, 'taiga_magazines': 0.8627, 'writers': 0.7443, 'Bible': 0.5099, 'political': 0.6232},
    'pikabu':          {'news': 0.6555, 'wikipedia': 0.7914, 'ficbook': 0.9428, 'pikabu': 1.0000, 'taiga_proza': 0.9381, 'taiga_magazines': 0.8902, 'writers': 0.7765, 'Bible': 0.5345, 'political': 0.7171},
    'taiga_proza':     {'news': 0.5734, 'wikipedia': 0.7523, 'ficbook': 0.9414, 'pikabu': 0.9381, 'taiga_proza': 1.0000, 'taiga_magazines': 0.9606, 'writers': 0.8521, 'Bible': 0.5651, 'political': 0.6980},
    'taiga_magazines': {'news': 0.6505, 'wikipedia': 0.8169, 'ficbook': 0.8627, 'pikabu': 0.8902, 'taiga_proza': 0.9606, 'taiga_magazines': 1.0000, 'writers': 0.8701, 'Bible': 0.5447, 'political': 0.6728},
    'writers':         {'news': 0.5271, 'wikipedia': 0.6901, 'ficbook': 0.7443, 'pikabu': 0.7765, 'taiga_proza': 0.8521, 'taiga_magazines': 0.8701, 'writers': 1.0000, 'Bible': 0.5859, 'political': 0.6245},
    'Bible':           {'news': 0.4305, 'wikipedia': 0.5123, 'ficbook': 0.5099, 'pikabu': 0.5345, 'taiga_proza': 0.5651, 'taiga_magazines': 0.5447, 'writers': 0.5859, 'Bible': 1.0000, 'political': 0.5059},
    'political':       {'news': 0.5698, 'wikipedia': 0.6615, 'ficbook': 0.6232, 'pikabu': 0.7171, 'taiga_proza': 0.6980, 'taiga_magazines': 0.6728, 'writers': 0.6245, 'Bible': 0.5059, 'political': 1.0000},
}

# Penalty-only weighting (2A.8.2, revised): the goal is "avoid target domains too
# CLOSE to the source", NOT "prefer the most distant". So every candidate sits at
# a uniform baseline (weight 1.0) and is *penalized* only when its centroid sim to
# the source exceeds TAU; the penalty ramps linearly to the floor at SIM_HI. This
# leaves distant/outlier registers (notably Bible, cos ~0.43-0.59) at ~uniform
# instead of hoarding the target budget (the earlier reward-distance formula gave
# Bible ~0.27), while still driving the genuinely near-register pairs (news<->wiki
# 0.965, ficbook<->pikabu 0.943, the prose cluster) toward the floor.
#   TAU=0.82   threshold on the raw centroid COSINE in DOMAIN_CENTROID_SIM (NOT
#              sim_measure): below = not too close = uniform baseline. Chosen
#              empirically — it cleanly separates the same-register/near-identity
#              cluster (informal prose + news<->wiki, all 0.86-0.965) from the
#              cross-register pairs and outliers (<=0.85). A reasonable data-driven
#              cut, not a first-principles constant; retune if the pool/encoder
#              changes (see target_pool_diagnostics.py for the matrix's origin).
#   SIM_HI=0.97 at/above this = maximal penalty (weight -> FLOOR); anchored to the
#              single tightest observed pair (news<->wikipedia 0.965), so the top
#              of the penalty ramp is effectively exercised by that one pair.
#   FLOOR=0.05 positivity floor: keeps every register reachable and guarantees
#              rng.choice(replace=False, p=) for n_picks=2 always has >=2 non-zero
#              entries. Post-renorm the smallest weight lands ~0.01 (a floor on the
#              pre-renorm weight, not a guaranteed minimum sampling probability).
TARGET_WEIGHT_FLOOR = 0.05
TARGET_WEIGHT_TAU = 0.82
TARGET_WEIGHT_SIM_HI = 0.97


def compute_domain_target_weights(source_domain, sim=DOMAIN_CENTROID_SIM,
                                  floor=TARGET_WEIGHT_FLOOR,
                                  tau=TARGET_WEIGHT_TAU,
                                  sim_hi=TARGET_WEIGHT_SIM_HI):
    """Per-source target-domain sampling weights (2A.8.2, penalty-only).

    For source ``s`` and each candidate target ``t != s``, start from a uniform
    baseline and apply a closeness penalty only above ``tau``::

        excess  = clip((sim(s, t) - tau) / (sim_hi - tau), 0, 1)
        w_{s,t} = 1 - (1 - floor) * excess

    then renormalize over the candidate set to a probability vector. Distant
    domains (``sim <= tau``) stay at the uniform baseline; near-register domains
    are suppressed toward ``floor``. This is applied only to single-domain
    (``other_domain``) target selection — 2-domain mixes stay uniform, because a
    mixture's closeness to the source is not monotonic in its components' (see
    ``add_target_style_emb``).

    Returns a ``{target_domain: weight}`` dict summing to 1 over all domains
    except ``source_domain``, or ``None`` if ``source_domain`` is unknown (caller
    then falls back to uniform sampling).
    """
    if source_domain not in sim:
        return None
    span = (sim_hi - tau) or 1.0
    raw = {}
    for t in sim:
        if t == source_domain:
            continue
        excess = min(max((sim[source_domain][t] - tau) / span, 0.0), 1.0)
        raw[t] = 1.0 - (1.0 - floor) * excess
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()}

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

def _rescale_targets_to_source_norms(source_embs, target_embs, target_norm=None):
    """Rescale each target embedding's norm.

    By default (``target_norm=None``) each target is scaled to the norm of its
    corresponding source embedding — the legacy behaviour for pre-folder-14
    checkpoints, whose style inputs sit at ~15 norm. Pass ``target_norm=1.0``
    for folder-14, which expects **unit-norm** style inputs (norm-fix Option 1,
    validated in task 2A.4 Phase A).
    """
    if target_norm is None:
        dst_norms = np.array([np.linalg.norm(s) or 1.0 for s in source_embs])
    else:
        dst_norms = np.full(len(source_embs), float(target_norm))
    tgt_norms = np.linalg.norm(target_embs, axis=1)
    tgt_norms_safe = np.where(tgt_norms == 0, 1.0, tgt_norms)
    scaled = target_embs * (dst_norms / tgt_norms_safe)[:, None]
    return [scaled[i] for i in range(scaled.shape[0])]

# ---------------- Main generic function ----------------

def produce_target_style(
    short_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    key_col: str,
    n_picks: int = 1,
    source_emb_col: str = "text_style_emb",
    rng: Optional[np.random.Generator] = None,
    target_norm: Optional[float] = None,
    key_weights: Optional[dict] = None,
) -> List[np.ndarray]:
    """
    For each row in short_df:
      - choose n_picks random *different* values of key_col from complete_df, excluding the row's own value
      - sample one embedding per chosen group
      - mix them using random weights (if n_picks > 1)
      - rescale to ``target_norm`` if given, else to the source embedding's norm

    ``key_weights`` optionally biases the *which-key* draw (2A.8.2): a
    ``{key_value: weight}`` map over candidate keys. When given, candidates are
    sampled with ``p`` proportional to these weights (renormalized over the
    candidate set after own-key exclusion), rather than uniformly. Keys missing
    from the map fall back to weight 0 among present ones — callers must supply a
    strictly-positive floor for every candidate so a probability vector always
    exists and ``replace=False`` for ``n_picks=2`` never hits "fewer non-zero
    entries than size". Passing ``key_weights`` changes ``Generator.choice``'s
    internal algorithm, so the seeded RNG bit stream differs from the unweighted
    path (reproducible, but re-baselined — see tests).

    The ``n_picks=2`` mix keeps the raw weighted sum (no per-pick unit
    normalization) — a deliberate choice preserved from the original mixing
    behaviour (task 2A.4 decision 3); the final ``_rescale_targets_to_source_norms``
    sets the norm.
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

        # Optional weighted draw over candidate keys (renormalized per row after
        # own-key exclusion). Uniform when key_weights is None.
        p = None
        if key_weights is not None:
            w = np.array([key_weights.get(k, 0.0) for k in candidates], dtype=float)
            total = w.sum()
            if total <= 0:
                p = None  # degenerate map for this candidate set → uniform
            else:
                p = w / total

        # Pick n_picks distinct or with replacement if not enough
        replace = len(candidates) < n_picks
        if p is not None and not replace and int((p > 0).sum()) < n_picks:
            # Sparse weight map: fewer non-zero candidates than n_picks would make
            # a without-replacement weighted draw raise "fewer non-zero entries in
            # p than size". The production floor keeps p fully positive so this
            # only guards misuse of the public wrappers — degrade to replacement.
            replace = True
        picks = rng.choice(candidates, size=n_picks, replace=replace, p=p)

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
    return _rescale_targets_to_source_norms(src_embs, target_embs, target_norm=target_norm)

# ---------------- Thin wrappers ----------------

def produce_target_style_random_writer(
    short_df, writers_df, author_col="author", source_emb_col="text_style_emb",
    rng=None, target_norm=None
):
    return produce_target_style(
        short_df, writers_df, key_col=author_col, n_picks=1,
        source_emb_col=source_emb_col, rng=rng, target_norm=target_norm
    )

def produce_target_style_2_random_writers(
    short_df, writers_df, author_col="author", source_emb_col="text_style_emb",
    rng=None, target_norm=None
):
    return produce_target_style(
        short_df, writers_df, key_col=author_col, n_picks=2,
        source_emb_col=source_emb_col, rng=rng, target_norm=target_norm
    )

def produce_target_style_other_domain(
    short_df, complete_df, domain_col="domain", source_emb_col="text_style_emb",
    rng=None, target_norm=None, domain_weights=None
):
    return produce_target_style(
        short_df, complete_df, key_col=domain_col, n_picks=1,
        source_emb_col=source_emb_col, rng=rng, target_norm=target_norm,
        key_weights=domain_weights,
    )

def produce_target_style_2_other_domains(
    short_df, complete_df, domain_col="domain", source_emb_col="text_style_emb",
    rng=None, target_norm=None, domain_weights=None
):
    return produce_target_style(
        short_df, complete_df, key_col=domain_col, n_picks=2,
        source_emb_col=source_emb_col, rng=rng, target_norm=target_norm,
        key_weights=domain_weights,
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
    
def get_target_styles(current_domain, has_author_targets=None):
    """Target-style-type list for a source domain.

    ``has_author_targets=None`` derives the flag from the 2A.8.1b ``DOMAIN_ONLY``
    set: domain-only domains (news/wikipedia/taiga_magazines/Bible/political) get
    domain targets only, real-author domains also get author targets. Callers
    (PphGenerator) normally pass the flag explicitly.
    """
    base_domain_target_style_types = ['other_domain', '2_domains_weighted_avg']
    base_author_target_style_types = ['other_author', '2_authors_weighted_avg']
    if has_author_targets is None:
        has_author_targets = current_domain not in DOMAIN_ONLY
    if not has_author_targets:
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

def add_target_style_emb(df, style_df, in_domain_style_df=None, target_norm=None,
                         rng=None, domain_weights=None):
    for target_style_desc in df.target_style_desc.unique():
        df_target_style = df[df.target_style_desc == target_style_desc]
        if not df_target_style.empty:
            if 'other_author' in target_style_desc:
                target_style_embs = produce_target_style_random_writer(
                    df_target_style, in_domain_style_df, rng=rng, target_norm=target_norm
                )
            elif '2_authors_weighted_avg' in target_style_desc:
                target_style_embs = produce_target_style_2_random_writers(
                    df_target_style, in_domain_style_df, rng=rng, target_norm=target_norm
                )
            elif 'other_domain' in target_style_desc:
                target_style_embs = produce_target_style_other_domain(
                    df_target_style, style_df, rng=rng, target_norm=target_norm,
                    domain_weights=domain_weights,
                )
            elif '2_domains_weighted_avg' in target_style_desc:
                # 2-domain mixes stay UNIFORM (domain_weights not forwarded): a
                # mixture's closeness to the source is not monotonic in its two
                # components' closeness, so per-pick weighting can't faithfully
                # control it. The per-pair SIM_MEASURE_TARGET_STYLE_UB filter
                # backstops any too-close mixture (2A.8.2 decision).
                target_style_embs = produce_target_style_2_other_domains(
                    df_target_style, style_df, rng=rng, target_norm=target_norm,
                )
            else:
                raise Exception('Unsupported target style type: ' + target_style_desc)
            # noise/negation are norm-preserving, so they do not disturb target_norm
            if '_with_noise' in target_style_desc:
                target_style_embs = add_noise_to_embs(target_style_embs, scale=0.01, rng=rng)
            elif '_with_negations' in target_style_desc:
                qt = 0.02 if 'author' in target_style_desc else 0.01
                target_style_embs = negate_some_vals(target_style_embs, quantile=qt, rng=rng)
            df.loc[df_target_style.index, 'target_style_emb'] = pd.Series(
                target_style_embs, index=df_target_style.index, dtype=object
            )

# Columns the keep-mask reads; perplexity/nat_v2 etc. must already be populated.
PHASE1_FILTER_COLS = [
    'chrf', 'bi_score', 'cl_score', 'gender_score', 'gender_activated',
    'entity_score', 'meaning_score', 'tst_result_style_sim', 'nat_v2',
]

def phase1_keep_mask(tst_res):
    """Boolean keep-mask for the A3-confirmed Phase-1 filter set (task 2A.4).

    Every threshold below is a REJECTION condition; the rows to KEEP are the
    negation. The gender gate fires only on activated pairs. ``tst_res`` must
    already carry every column in ``PHASE1_FILTER_COLS``.

    Factored out of ``gen_paraphrases`` so the threshold logic is unit-testable
    without a GPU. NaN in a filter column would silently drop the row, so this
    fails loud instead — a scorer regression (e.g. empty ``styled_text``)
    should surface as an error, not a quietly lower pass rate.
    """
    missing = [c for c in PHASE1_FILTER_COLS if c not in tst_res.columns]
    if missing:
        raise KeyError(f"phase1_keep_mask: missing filter columns: {missing}")
    # gender_score may legitimately be NaN on non-activated rows (it is masked
    # out there); only flag NaN among the rows the gender gate actually reads.
    plain = [c for c in PHASE1_FILTER_COLS
             if c not in ('gender_score', 'gender_activated')]
    nan_cols = [c for c in plain if tst_res[c].isna().any()]
    if (tst_res.gender_activated.astype(bool) & tst_res.gender_score.isna()).any():
        nan_cols.append('gender_score(activated)')
    if nan_cols:
        raise ValueError(f"phase1_keep_mask: NaN in filter columns: {nan_cols}")
    return (
        (tst_res.chrf < CHRF_UB) &
        (tst_res.bi_score < BI_SCORE_UB) &
        (tst_res.cl_score < CL_SCORE_UB) &
        ~(tst_res.gender_activated.astype(bool) & (tst_res.gender_score < GENDER_SCORE_LB)) &
        (tst_res.entity_score >= ENTITY_SCORE_LB) &
        (tst_res.meaning_score > MEANING_SCORE_LB) &
        (tst_res.tst_result_style_sim < SIM_MEASURE_UB) &
        (tst_res.nat_v2 >= NAT_V2_LB)
    )

def gen_paraphrases(
    current_df, tst_generator, style_df, in_domain_style_df, target_styles,
    alignment_scorer, gender_scorer, entity_scorer,
    target_norm=None, perplexity_batch_size=PERPLEXITY_BATCH_SIZE, rng=None,
    domain_weights=None,
):
    current_df = current_df.copy()
    if rng is None:
        rng = np.random.default_rng()
    current_df['target_style_desc'] = rng.choice(
        target_styles, size=current_df.shape[0]
    )
    add_target_style_emb(
        current_df, style_df, in_domain_style_df, target_norm=target_norm, rng=rng,
        domain_weights=domain_weights,
    )
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
        verbose=True,
        perplexity_batch_size=perplexity_batch_size,
    )
    # Canonical scoring order (mirrors the eval framework, minus score_v2):
    #   compute_scores -> compute_quality_scores -> select_best -> copying metrics.
    # quality scores run on *all* versions (before selection) so they survive into
    # best_tst_results; perplexity uses the reduced batch size to fit 8 GB.
    pm.add_source_ppx_and_emb()
    pm.produce_tst_results()
    pm.compute_scores()
    pm.compute_quality_scores(alignment_scorer, gender_scorer, entity_scorer)
    pm.select_best_tst_version()
    pm.compute_copying_metrics()  # chrf on best_tst_results
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
    tst_res['nat_v2'] = compute_nat_v2(
        tst_res.styled_text_perplexity, tst_res.text_perplexity, tst_res.target_style_desc
    )
    return tst_res[phase1_keep_mask(tst_res)]

# Columns that must reach disk for a survivor to be auditable downstream; if any
# is absent from the results frame, the scoring order upstream is broken — fail
# loud rather than silently dropping it via the whitelist intersection.
REQUIRED_PERSIST_COLS = [
    'styled_text', 'styled_text_style_emb', 'meaning_score',
    'tst_result_style_sim', 'bi_score', 'cl_score', 'gender_score',
    'entity_score', 'chrf', 'nat_v2',
]

def save_tst_results(tst_df, results_path):
    tst_res_cols = [
        'text', 'target_style_desc', 'target_style_sim_measure', 'styled_text',
        'styled_text_style_emb', 'meaning_score',
        'tst_result_style_sim',
        # Phase-1 quality signals persisted for downstream auditing (2A.4):
        'bi_score', 'cl_score', 'gender_score', 'gender_activated',
        'entity_score', 'chrf', 'nat_v2',
        # naturality_score is LEGACY — kept for reference only; it is NOT a gate
        # in the 2A.4 pipeline (mis-calibrated for folder-14, replaced by nat_v2).
        'naturality_score', 'style_score', 'score',
    ]
    missing = [c for c in REQUIRED_PERSIST_COLS if c not in tst_df.columns]
    if missing:
        raise KeyError(f"save_tst_results: required columns missing: {missing}")
    tst_res_cols = [c for c in tst_res_cols if c in tst_df.columns]
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
    tst_df['styled_text_style_emb'] = tst_df['styled_text_style_emb'].map(
        lambda se: se.astype(np.float16)
    )
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
        checkpoint_path,
        long_texts=False,
        rows_at_once=30000,
        base_model_path = "ai-forever/rugpt3small_based_on_gpt2",
        assert_norm=None,
        seed=DEFAULT_SEED,
        perplexity_batch_size=PERPLEXITY_BATCH_SIZE,
    ):
        # `assert_norm` is threaded through to the inner TinyStyler. Default
        # None because PphGenerator is used with both folder-14 (unit-norm)
        # and pre-folder-14 (~15 norm) checkpoints — the caller must pick the
        # value that matches the checkpoint they pass in.
        set_global_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.perplexity_batch_size = perplexity_batch_size
        # Norm-fix Option 1: folder-14 (`assert_norm='normalized'`) expects
        # unit-norm target style embeddings; older checkpoints keep the
        # source-norm rescale (target_norm=None).
        self.target_norm = 1.0 if assert_norm == 'normalized' else None
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_df = pd.read_parquet(base_df_path)
        self.style_df = pd.read_parquet(style_df_path)
        self.results_path = results_path
        self.rows_at_once = rows_at_once
        # 2A.8.2: fresh-encode the target style pool from its text rather than
        # trusting a stored embedding column. style_sample_v2 ships text only
        # ([author, domain, text_chunk], no text_style_emb), and legacy pools had
        # a ~10% text<->embedding desync — re-encoding here guarantees the target
        # geometry matches the pool text. Write into `text_style_emb` because
        # produce_target_style groups the pool on that column. normalize=False
        # keeps the source-norm convention; target_norm handles any rescale.
        pool_text_col = 'text_chunk' if 'text_chunk' in self.style_df.columns else 'text'
        self.style_df['text_style_emb'] = calc_style_embeddings(
            self.style_df[pool_text_col].tolist(), normalize=False
        )
        # calc_style_embeddings builds a fresh SentenceTransformer and does not
        # free it; release the VRAM before TinyStyler + the 3 resident scorers
        # load onto tallin's ~8 GB card.
        gc.collect()
        torch.cuda.empty_cache()
        current_domain = self.base_df.iloc[0].domain
        in_domain = self.style_df[self.style_df.domain == current_domain]
        # Per-source target-domain weights (2A.8.2): down-weight near-neighbor
        # target registers. None for unknown domains -> uniform sampling.
        self.domain_weights = compute_domain_target_weights(current_domain)
        # 2A.8.1b: domain-only domains (news/wikipedia/taiga_magazines/Bible/
        # political) get domain-only target styles; real-author domains also get
        # author targets. Use the explicit DOMAIN_ONLY set — NOT in_domain.empty,
        # which flips to author-targets once author-less domains carry
        # author=<domain> rows in style_sample_v2.
        has_author_targets = current_domain not in DOMAIN_ONLY
        self.target_styles = get_target_styles(
            current_domain, has_author_targets=has_author_targets)
        self.in_domain_style_df = in_domain if has_author_targets else None
        if long_texts:
            max_length = 1024
            batch_size = 6
        else:
            max_length = 256
            batch_size = 10
        model = TinyStyler(
            model_name=base_model_path, model_type='GPT', use_style=True,
            checkpoint_path=checkpoint_path,
            assert_norm=assert_norm,
        ).cuda()
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
            max_input_length=max_length,
            max_output_length=max_length,
        )
        # Phase-1 filter scorers, constructed once and kept warm across chunks
        # (reloading mBERT/NER per chunk would dominate runtime). VRAM: these sit
        # resident alongside the generator (~3 GB total); the only transient peak
        # is batch-8 perplexity (~1.6 GB) — comfortably under tallin's ~8 GB.
        # mBERT span4 (task 2A.3.1); gender uses mBERT internally (task 1.6);
        # entity is NER-only (no LaBSE/BiCl/pre_clean — defaults).
        self.alignment_scorer = BatchedAligner(
            model='bert', score_fn='span', min_span=4, device='cuda'
        )
        self.gender_scorer = GenderConsistencyScorer(device='cuda')
        self.entity_scorer = EntityConsistencyScorer(device='cuda')

    def execute(self, max_attempts=3, prefetch=True):
        """Stream the base pool through generation+filtering.

        The queue is shuffled once; failures go to the back (FIFO) and are
        retried with a freshly sampled target style (a row that fails one target
        often passes another). ``max_attempts`` caps retries so the hard tail is
        abandoned rather than recirculated forever — source is abundant, compute
        is the bottleneck (task 2A.4 A4 / B5 finding). ``prefetch`` caches the
        source-side features (perplexity/style/LaBSE) on the base pool **once**
        so requeued rows don't recompute them every attempt (B4; the per-attempt
        source recompute was ~55 ms/row × the retry redundancy).
        """
        if prefetch:
            # Idempotent: text_style_emb already lives on the pools, so only
            # perplexity + LaBSE are computed here, once for the whole pool.
            ensure_source_caches(
                self.base_df, perplexity=True, style_emb=True, labse_emb=True,
                perplexity_batch_size=self.perplexity_batch_size,
            )
            gc.collect()
            torch.cuda.empty_cache()
        generated_df = join_generated_files(self.results_path)
        processed_ids = generated_df.index if (generated_df is not None) else []
        unprocessed_queue = list(set(self.base_df.index) - set(processed_ids))
        # Shuffle via the instance rng (seeded in __init__) so the run is
        # reproducible from `seed` alone, not the global numpy state.
        unprocessed_queue = list(self.rng.permutation(unprocessed_queue))
        attempts = {i: 0 for i in unprocessed_queue}
        n_abandoned = 0
        while unprocessed_queue:
            time_start = datetime.now()
            processing_ids = unprocessed_queue[:self.rows_at_once]
            unprocessed_queue = unprocessed_queue[self.rows_at_once:]
            current_df = self.base_df.loc[processing_ids]
            tst_df = gen_paraphrases(
                current_df, self.tst_generator,
                self.style_df, self.in_domain_style_df,
                self.target_styles,
                self.alignment_scorer, self.gender_scorer, self.entity_scorer,
                target_norm=self.target_norm,
                perplexity_batch_size=self.perplexity_batch_size,
                rng=self.rng,
                domain_weights=self.domain_weights,
            )
            save_tst_results(tst_df, self.results_path)
            completed_ids = set(tst_df.index)
            requeued_ids = []
            for i in processing_ids:
                if i in completed_ids:
                    continue
                attempts[i] += 1
                if attempts[i] < max_attempts:
                    requeued_ids.append(i)   # rejects go to the back of the queue
                else:
                    n_abandoned += 1          # capped out — abandon (source is abundant)
            unprocessed_queue += requeued_ids
            # Release this chunk's cached GPU blocks before the next one — the
            # transient scoring models otherwise fragment the small (~8 GB) GPU
            # across the streaming requeue loop and OOM after a few chunks.
            del tst_df, current_df
            gc.collect()
            torch.cuda.empty_cache()
            time_end = datetime.now()
            time_str = time_end.strftime('%Y-%m-%d %H:%M:%S')
            n_tried = len(processing_ids)
            n_succeeded = len(completed_ids)
            n_unprocessed = len(unprocessed_queue)
            n_total = self.base_df.shape[0]
            success_p = f"{n_succeeded/n_tried:.1%}" if n_tried else "n/a"
            print(
                f"{n_succeeded} from {n_tried} ({success_p}) generated paraphrases passed during last step."
            )
            print(f"Step time: {str(time_end-time_start)}")
            print(
                f"Totally: {time_str} | queue {n_unprocessed} | abandoned {n_abandoned} "
                f"(cap {max_attempts}) | pool {n_total}"
            )