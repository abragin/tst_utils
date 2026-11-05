"""Per-metric base-score helpers (operate in place on the results DataFrame).

Each function adds one metric family's columns to ``df`` (the expanded
tst_results). ``log`` is an optional progress callback ``(msg, *args) -> None``
(defaults to a no-op) so the orchestrator can route verbose progress through its
logger without these functions knowing about it.
"""

import gc

import numpy as np
import torch
from tst_utils.eval.metrics.naturality import calculate_perplexity, naturality_score
from tst_utils.eval.metrics.meaning import (
    meaning_score, b_score, labse_scores_from_embs,
    calc_labse_embeddings, length_penalty
)
from tst_utils.eval.metrics.style import calc_style_embeddings, add_away_towards
from tst_utils.eval.data.style_store import StyleLabelError
from tst_utils.eval.performance.source_cache import (
    encode_style_rows, ensure_source_caches)


def _noop(*args) -> None:
    pass


def _free_gpu() -> None:
    """Release the just-used transient model's cached GPU blocks. Each base-score
    step loads a model, scores, and discards it; without this the CUDA caching
    allocator fragments across a long streaming run and eventually OOMs on a
    small GPU (tallin ~8 GB). expandable_segments is not an option here (it
    crashes this box's driver — see task 2A.4 A1), so we defragment explicitly."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_perplexity_scores(df, log=_noop, batch_size=32) -> None:
    log('Adding perplexity')
    ensure_source_caches(df, perplexity=True, perplexity_batch_size=batch_size)
    df['styled_text_perplexity'] = calculate_perplexity(
        df.styled_text, batch_size=batch_size)[0]
    _free_gpu()


def add_bert_score(df, log=_noop) -> None:
    log('Adding bert score')
    df['bert_score'] = b_score(df.text, df.styled_text)
    _free_gpu()


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
    _free_gpu()


def add_style_scores(df, author_styles, log=_noop) -> None:
    """Historical route, kept for the frozen phase-2A callers
    (`analyze_hard.py`, `gen_phaseb_named.py`, `gen_static_set.py`,
    `salvage_static_set.py`, `score_static_set.py`); it encodes with the
    pin. `TstPerformanceMetrics` uses `add_labelled_style_scores`."""
    log('Adding style embeddings')
    ensure_source_caches(df, style_emb=True)
    df['styled_text_style_emb'] = calc_style_embeddings(df.styled_text, normalize=False)
    add_away_towards(df, author_styles)
    _free_gpu()


def add_labelled_style_scores(df, targets, *, key_column, source_vectors,
                              encoder_key, log=_noop) -> None:
    """Style scores against a labelled target source (task 5.2).

    `styled_text` is encoded with `encoder_key` (the instance's evaluation
    encoder); `source_vectors` are the source vectors in row order, taken
    from the instance's store; each row's target is the unit view of
    `targets.vector(row[key_column])`.
    """
    if targets.encoder_key != encoder_key:
        raise StyleLabelError(
            f"add_labelled_style_scores: the targets hold "
            f"{targets.encoder_key!r} vectors; the styled texts would be "
            f"encoded with {encoder_key!r}")
    log('Adding style embeddings')
    df['styled_text_style_emb'] = encode_style_rows(df.styled_text, encoder_key)
    add_away_towards(df, targets, key_column=key_column,
                     source_vectors=source_vectors)
    _free_gpu()


def combine_scores(df, log=_noop) -> None:
    log('Adding scores')
    df['style_score'] = np.sqrt(df.away * df.towards)
    length_penalties = length_penalty(df.text, df.styled_text)
    df['meaning_score'] = meaning_score(df.bert_score, df.labse_score, length_penalties)
    df['naturality_score'] = naturality_score(df.text_perplexity, df.styled_text_perplexity)
    df['score'] = (
        df['style_score'] * df['meaning_score'] * df['naturality_score']
    ) ** (1. / 3)
