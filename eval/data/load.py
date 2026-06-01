import pandas as pd
import numpy as np
import os


def load_test_df(short=False):
    """Load the main test dataset"""
    if short:
        file_path = os.path.join(os.path.dirname(__file__), "test_data_short.parquet.gzip")
    else:
        file_path = os.path.join(os.path.dirname(__file__), "test_data.parquet.gzip")
    return pd.read_parquet(file_path)

def load_author_styles():
    """Load vector representations for main target styles.

    Returns the canonical `author_styles.npz` centroids as a dict.
    These centroids are **unnormalized** (norm ~15) — the pre-folder-14
    scale. Pass them to TinyStyler with ``assert_norm='unnormalized'``.
    """
    file_path = os.path.join(os.path.dirname(__file__), "author_styles.npz")
    with np.load(file_path) as loaded:
        author_styles = {key: loaded[key] for key in loaded}
    return author_styles


def renormalize_centroid(arr):
    """L2-normalize a centroid (or batch of centroids) for use with TinyStyler.

    Centroids computed as per-component means of unit-norm chunk embeddings
    have ``||mean|| ≤ 1.0`` by Jensen's inequality — typically 0.88–0.99,
    depending on intra-cluster tightness. TinyStyler folder-14 was trained
    on unit-norm style inputs, so feeding raw (sub-unit) centroids pushes
    the projected style signal off-distribution. Renormalize before passing.

    Accepts a 1D vector (single centroid) or a 2D batch ``(n, dim)``.
    See ``docs/issues/centroid-renormalization.md`` for context.
    """
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


# Minimum vector size for which `load_centroids_npz` will apply renormalize.
# Anything smaller is treated as metadata (e.g. small ID/label arrays) and
# returned untouched. 50 picks every realistic embedding dim (`abragin/ruBert-style-base`
# is 768) while still skipping incidental short float arrays.
_MIN_RENORMALIZE_SIZE = 50


def load_centroids_npz(path, *, renormalize):
    """Load centroid vectors from an `.npz` file.

    Args:
        path: path to a `.npz` file whose entries are 1D vectors (or 2D
            batches of vectors). Non-float entries (metadata) are returned
            untouched.
        renormalize: REQUIRED keyword. If True, every float-typed vector
            entry is L2-normalized to unit norm via
            :func:`renormalize_centroid`. Choose ``True`` when the loaded
            centroids will be passed to ``TinyStyler(assert_norm='normalized')``;
            choose ``False`` for cosine-only / statistical use.

    Returns:
        dict[str, np.ndarray]
    """
    with np.load(path) as loaded:
        result = {k: loaded[k] for k in loaded.files}
    if renormalize:
        result = {
            k: (renormalize_centroid(v)
                if (v.dtype.kind == 'f' and v.ndim >= 1 and v.size >= _MIN_RENORMALIZE_SIZE)
                else v)
            for k, v in result.items()
        }
    return result

def load_llm_data():
    """Load TST results performed by LLMs (manually)."""
    file_path = os.path.join(
        os.path.dirname(__file__), "llms_with_scores.csv.gz"
    )
    return pd.read_csv(file_path, compression='gzip')