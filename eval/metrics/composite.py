"""Composite score v2: base_score × nat_v2.

Formula
-------
meaning_base = meaning_nolen
gender_adj   = 0.10 + 0.90 × gender_score
entity_adj   = 0.10 + 0.90 × entity_score
# meaning_v2: WGM with weights (0.5, 0.5, 1.0, 0.5) summing to 2.5
meaning_v2   = (meaning_base^0.5 × bi_cl^0.5 × gender_adj^1.0 × entity_adj^0.5)^(1/2.5)
# base_score: WGM of style (weight 1.0) and meaning_v2 (weight 2.5), total 3.5
base_score   = (style^1.0 × meaning_v2^2.5)^(1/3.5)

tgt_ce_ref = max(author_lookup(target_style), source_CE)  # source_CE is always a floor
style_gap  = styled_CE − tgt_ce_ref
nat_v2     = 1 / (max(0, style_gap − 2.0) + 1)

score_v2   = base_score × nat_v2

Author/domain CE baselines are medians of CE(literary original text) from
rugpt3small, computed by 16 phase 2A infrastructure/recompute_ce.py
(N=500 samples/key, seed 1991, batch-invariant right-padded calculate_perplexity).
AUTHOR_CE basis: data/styled_pph/train_short.parquet.gzip `text`; DOMAIN_CE basis:
data/sources_v2/final/<domain>_train_v2.parquet `text` (9-domain pool_v2 taxonomy).
Domain medians are the fallback when target_style is not in AUTHOR_CE.

NOTE (2026-06-21): the previous values were corrupted by a left-padding bug in
calculate_perplexity that made CE batch-composition dependent (see
docs/issues/resolved/calculate-perplexity-left-pad-batch-dependence.md). These corrected
values are ~2 CE units lower and on a compressed scale (span ~0.9). The nat_v2
`margin` (2.0) was tuned on the old inflated scale and now leaves nat_v2 ~ 1.0 for
almost all inputs; it should be recalibrated on this scale to regain discrimination.
"""

import numpy as np
import pandas as pd

AUTHOR_CE = {
    'Chekhov':        3.641447,
    'Zoschenko':      3.813564,
    'News':           3.147179,
    'Dovlatov':       3.758374,
    'Putin':          3.414283,
    'Ilf and Petrov': 3.715437,
    'Medvedev':       3.643130,
    'Pushkin':        3.867682,
    'Lermontov':      3.768750,
    'Zhvaneckiy':     3.911052,
    'Bulgakov':       3.771773,
    'Dostoevsky':     3.835618,
    'Bryusov':        3.702599,
    'Yerofeyev':      3.755454,
    'Gorky':          3.853821,
    'Turgenev':       3.777817,
    'Blok':           3.897557,
    'Bible':          3.666696,
    'Tolstoy':        3.743540,
    'Gogol':          3.965799,
    'Herzen':         4.047286,
    'Nabokov':        3.930439,
}

# 9-domain pool_v2 taxonomy (2A.8.1b). Fallback when target_style not in AUTHOR_CE.
# NOTE: AUTHOR_CE and DOMAIN_CE sit on *different* text bases (AUTHOR_CE: train_short
# literary originals; DOMAIN_CE: data/sources_v2/final pools), so a key present in both
# ('Bible') has two slightly different values (author 3.667 vs domain 3.857) and the
# author entry wins via _lookup_target_ce. Also note this table is only consulted on the
# *eval* path (target_style); styled_pph generation passes target_style_desc (synthetic
# recipe strings, e.g. 'other_author'), which never match a key, so generation-time nat_v2
# is always source-referenced. See docs/issues/resolved/calculate-perplexity-left-pad-batch-dependence.md.
DOMAIN_CE = {
    'news':            3.151731,
    'wikipedia':       3.421043,
    'ficbook':         3.807949,
    'pikabu':          4.046271,
    'taiga_proza':     3.876610,
    'taiga_magazines': 3.876763,
    'writers':         3.962501,
    'Bible':           3.857263,
    'political':       3.649666,
}


def _lookup_target_ce(target_style: str):
    """Return author or domain median CE, or None if unknown."""
    v = AUTHOR_CE.get(target_style)
    return v if v is not None else DOMAIN_CE.get(target_style)


def base_score_v2(
    style_score, meaning_nolen, bi_cl, gender_score, entity_score, floor=0.10
):
    """Compute base composite score (without nat_v2 penalty).

    All inputs may be scalars or numpy arrays. Returns scalar or array.
    """
    s = np.asarray(style_score, float)
    m = np.asarray(meaning_nolen, float)
    b = np.asarray(bi_cl, float)
    g = np.asarray(gender_score, float)
    e = np.asarray(entity_score, float)
    gender_adj   = floor + (1.0 - floor) * g
    entity_adj   = floor + (1.0 - floor) * e
    meaning_v2   = (m ** 0.5 * b ** 0.5 * gender_adj ** 1.0 * entity_adj ** 0.5) ** (1.0 / 2.5)
    return (s ** 1.0 * meaning_v2 ** 2.5) ** (1.0 / 3.5)


def nat_v2_score(
    styled_ce: float, source_ce: float, target_style: str = '', margin: float = 2.0
) -> float:
    """Compute nat_v2 for a single example.

    tgt_ce_ref = max(author_lookup(target_style), source_ce) so that source_ce
    always acts as a floor. Falls back to source_ce when target_style is unknown.
    """
    lookup = _lookup_target_ce(target_style) if isinstance(target_style, str) else None
    tgt_ce_ref = max(lookup, source_ce) if lookup is not None else source_ce
    style_gap = styled_ce - tgt_ce_ref
    return 1.0 / (max(0.0, style_gap - margin) + 1.0)


def compute_nat_v2(styled_ce, source_ce, target_style_iter, margin: float = 2.0):
    """Vectorised nat_v2 over parallel iterables of CE values and target style names.

    Parameters
    ----------
    styled_ce, source_ce : array-like of float
    target_style_iter    : iterable of str — target author/style names; NaN/None → unknown
    """
    return np.array([
        nat_v2_score(
            float(sty), float(src),
            ts if isinstance(ts, str) else '',
            margin,
        )
        for sty, src, ts in zip(styled_ce, source_ce, target_style_iter)
    ])
