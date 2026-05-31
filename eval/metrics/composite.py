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

Author CE baselines are medians of CE(literary original text) from styled_pph,
computed on 25 samples per author using rugpt3small. Domain medians are used for
ficbook/news/writers when the target_style is not in AUTHOR_CE.
"""

import numpy as np
import pandas as pd

AUTHOR_CE = {
    'Chekhov':        4.965303,
    'Zoschenko':      5.025889,
    'News':           5.109265,
    'Dovlatov':       5.258266,
    'Putin':          5.313718,
    'Ilf and Petrov': 5.407140,
    'Medvedev':       5.416350,
    'Pushkin':        5.535736,
    'Lermontov':      5.541788,
    'Zhvaneckiy':     5.572086,
    'Bulgakov':       5.757597,
    'Dostoevsky':     5.850113,
    'Bryusov':        5.891932,
    'Yerofeyev':      5.897355,
    'Gorky':          5.959240,
    'Turgenev':       5.991673,
    'Blok':           6.050667,
    'Bible':          6.123743,
    'Tolstoy':        6.637148,
    'Gogol':          6.842763,
    'Herzen':         6.905677,
    'Nabokov':        8.702062,
}

DOMAIN_CE = {
    'ficbook': 5.573,
    'news':    5.109265,
    'writers': 5.725,
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
