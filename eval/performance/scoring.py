"""Per-metric base-score helpers (operate in place on the results DataFrame).

Each function adds one metric family's columns to ``df`` (the expanded
tst_results). ``log`` is an optional progress callback ``(msg, *args) -> None``
(defaults to a no-op) so the orchestrator can route verbose progress through its
logger without these functions knowing about it.
"""

import numpy as np
from tst_utils.eval.metrics.naturality import calculate_perplexity, naturality_score
from tst_utils.eval.metrics.meaning import (
    meaning_score, b_score, labse_scores_from_embs,
    calc_labse_embeddings, length_penalty
)
from tst_utils.eval.metrics.style import calc_style_embeddings, add_away_towards
from tst_utils.eval.performance.source_cache import ensure_source_caches


def _noop(*args) -> None:
    pass


def add_perplexity_scores(df, log=_noop) -> None:
    log('Adding perplexity')
    ensure_source_caches(df, perplexity=True)
    df['styled_text_perplexity'] = calculate_perplexity(df.styled_text)[0]


def add_bert_score(df, log=_noop) -> None:
    log('Adding bert score')
    df['bert_score'] = b_score(df.text, df.styled_text)


def add_labse_score(df, log=_noop) -> None:
    log('Adding labse score')
    # source LaBSE: use the cached column if present, else compute transiently
    # (not cached here — only add_source_ppx_and_emb caches it on test_df).
    if 'text_labse_emb' in df:
        text_labse_embs = df.text_labse_emb
    else:
        text_labse_embs = calc_labse_embeddings(df.text)
    styled_text_labse_embs = calc_labse_embeddings(df.styled_text)
    df['labse_score'] = labse_scores_from_embs(text_labse_embs, styled_text_labse_embs)


def add_style_scores(df, author_styles, log=_noop) -> None:
    log('Adding style embeddings')
    ensure_source_caches(df, style_emb=True)
    df['styled_text_style_emb'] = calc_style_embeddings(df.styled_text, normalize=False)
    add_away_towards(df, author_styles)


def combine_scores(df, log=_noop) -> None:
    log('Adding scores')
    df['style_score'] = np.sqrt(df.away * df.towards)
    length_penalties = length_penalty(df.text, df.styled_text)
    df['meaning_score'] = meaning_score(df.bert_score, df.labse_score, length_penalties)
    df['naturality_score'] = naturality_score(df.text_perplexity, df.styled_text_perplexity)
    df['score'] = (
        df['style_score'] * df['meaning_score'] * df['naturality_score']
    ) ** (1. / 3)
