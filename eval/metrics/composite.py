"""Composite score v2: base_score × nat_v2.

Formula
-------
entity_s   = 0.10 + 0.90 × entity_score
gender_s   = 0.10 + 0.90 × gender_score
meaning_v2 = sqrt(meaning_nolen × bi_cl)
base_score = (style × meaning_v2 × gender_s × entity_s^0.5)^(1/3.5)

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
    'Chekhov':        4.965,
    'Zoschenko':      5.026,
    'News':           5.109,
    'Dovlatov':       5.258,
    'Putin':          5.314,
    'Ilf and Petrov': 5.407,
    'Medvedev':       5.416,
    'Pushkin':        5.536,
    'Lermontov':      5.542,
    'Zhvaneckiy':     5.572,
    'Bulgakov':       5.758,
    'Dostoevsky':     5.850,
    'Bryusov':        5.892,
    'Yerofeyev':      5.897,
    'Gorky':          5.959,
    'Turgenev':       5.992,
    'Blok':           6.051,
    'Bible':          6.124,
    'Tolstoy':        6.637,
    'Gogol':          6.843,
    'Herzen':         6.906,
    'Nabokov':        8.702,
}

DOMAIN_CE = {
    'ficbook': 5.573,
    'news':    5.109,
    'writers': 5.725,
}


def _lookup_target_ce(target_style: str):
    """Return author or domain median CE, or None if unknown."""
    return AUTHOR_CE.get(target_style) or DOMAIN_CE.get(target_style)


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
    entity_s  = floor + (1.0 - floor) * e
    gender_s  = floor + (1.0 - floor) * g
    meaning_v2 = np.sqrt(m * b)
    return (s * meaning_v2 * gender_s * entity_s ** 0.5) ** (1.0 / 3.5)


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
            ts if (isinstance(ts, str) or pd.notna(ts)) and isinstance(ts, str) else '',
            margin,
        )
        for sty, src, ts in zip(styled_ce, source_ce, target_style_iter)
    ])
