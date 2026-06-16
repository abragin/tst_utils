"""Source-side cache population (perplexity / style / LaBSE embeddings).

Single source of truth for the source-text feature formulas, shared by the
source frame (``add_source_ppx_and_emb``) and the expanded results frame
(``compute_scores``). Each cache is only (re)computed when its column is absent.
"""

from tst_utils.eval.metrics.naturality import calculate_perplexity
from tst_utils.eval.metrics.meaning import calc_labse_embeddings
from tst_utils.eval.metrics.style import calc_style_embeddings


def ensure_source_caches(df, *, perplexity=False, style_emb=False, labse_emb=False,
                         perplexity_batch_size=32) -> None:
    """Idempotently populate the requested source-side caches on ``df`` (computed
    from ``df.text``). Only the caches whose flag is set are touched, and only
    when the corresponding column is absent. ``perplexity_batch_size`` controls
    the rugpt3 perplexity batch size (lower it to fit a small GPU — its per-batch
    logits tensor dominates VRAM; see styled_pph_gen's 8 GB path).

    CAVEAT — a stored cache column is TRUSTED VERBATIM (compute-if-absent): if
    ``df`` already carries e.g. ``text_style_emb``, it is used as-is and NOT
    re-derived from ``df.text``, so a stale/desynced stored vector silently wins
    over a fresh recompute. Some legacy pools ship a ``text_style_emb`` that does
    not match their own ``text`` (see
    the ficbook-text_style_emb-desync task in the main repo's docs/tasks/complete/). When feeding a
    pool of uncertain provenance, DROP the cache column first so it is recomputed
    fresh from ``text`` (``text`` is the ground truth)."""
    if perplexity and 'text_perplexity' not in df:
        df['text_perplexity'] = calculate_perplexity(
            df.text, batch_size=perplexity_batch_size)[0]
    if style_emb and 'text_style_emb' not in df:
        df['text_style_emb'] = calc_style_embeddings(df.text, normalize=False)
    if labse_emb and 'text_labse_emb' not in df:
        df['text_labse_emb'] = [
            e.cpu().numpy() for e in calc_labse_embeddings(df.text)
        ]
