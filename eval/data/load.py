import json

import pandas as pd
import numpy as np
import os

from tst_utils.eval.data.style_store import StyleLabelError, warn_provenance
from tst_utils.eval.model_names import STYLE_ENCODER_KEY
from tst_utils.eval.style_encoder_registry import get_encoder


def load_test_df(short=False):
    """Load the main test dataset"""
    if short:
        file_path = os.path.join(os.path.dirname(__file__), "test_data_short.parquet.gzip")
    else:
        file_path = os.path.join(os.path.dirname(__file__), "test_data.parquet.gzip")
    return pd.read_parquet(file_path)

def read_style_label(path):
    """Return the label dict of an npz file, or None for an unlabelled file."""
    with np.load(path) as loaded:
        if "__style_label__" not in loaded.files:
            return None
        return json.loads(str(loaded["__style_label__"]))


def _check_npz_label(path, label, expect_encoder, entry_point):
    if label is None:
        if expect_encoder == "base_v1":
            warn_provenance(
                f"{entry_point}: unlabelled npz {os.path.realpath(path)}; "
                "an unlabelled file predates the label; a newer unlabelled "
                "file must be labelled, not loaded"
            )
            return
        raise StyleLabelError(
            f"{entry_point}: unlabelled npz {os.path.realpath(path)}, "
            f"expected encoder {expect_encoder!r}"
        )
    if label.get("encoder_key") != expect_encoder:
        raise StyleLabelError(
            f"{entry_point}: npz {path} names encoder "
            f"{label.get('encoder_key')!r}, expected {expect_encoder!r}"
        )
    if label.get("file_sha256") != get_encoder(expect_encoder)["file_sha256"]:
        raise StyleLabelError(
            f"{entry_point}: npz {path} names a file_sha256 that differs "
            f"from the registry entry of {expect_encoder!r}"
        )


def load_author_styles(expect_encoder=None):
    """Load vector representations for main target styles.

    Returns the canonical `author_styles.npz` centroids as a dict.
    These centroids are **unnormalized** (norm ~15) — the pre-folder-14
    scale. Pass them to TinyStyler with ``assert_norm='unnormalized'``.

    The `__style_label__` entry is removed, so the return type stays the
    same. `expect_encoder` defaults to the pin: an unlabelled file loads
    with a warning while the pin is `base_v1`, and raises otherwise.
    A filename is never a label.
    """
    if expect_encoder is None:
        expect_encoder = STYLE_ENCODER_KEY
    file_path = os.path.join(os.path.dirname(__file__), "author_styles.npz")
    _check_npz_label(file_path, read_style_label(file_path), expect_encoder,
                     "load_author_styles")
    with np.load(file_path) as loaded:
        author_styles = {key: loaded[key] for key in loaded if key != "__style_label__"}
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


def load_centroids_npz(path, *, renormalize, expect_encoder=None):
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

    The `__style_label__` entry is removed, so the return type stays the
    same. `expect_encoder` defaults to the pin: an unlabelled file loads
    with a warning while the pin is `base_v1`, and raises otherwise.
    A filename is never a label.
    """
    if expect_encoder is None:
        expect_encoder = STYLE_ENCODER_KEY
    _check_npz_label(path, read_style_label(path), expect_encoder,
                     "load_centroids_npz")
    with np.load(path) as loaded:
        result = {k: loaded[k] for k in loaded.files if k != "__style_label__"}
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