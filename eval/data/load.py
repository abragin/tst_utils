import json

import pandas as pd
import numpy as np
import os

from tst_utils.eval.data.style_store import (
    CENTROID_LABEL_FIELDS,
    StyleLabelError,
    require_label_fields,
    warn_provenance,
)
from tst_utils.eval import model_names
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


# Minimum element count for a float array to count as a centroid vector.
# Anything smaller is metadata (e.g. a short id/label array) and is neither
# renormalized nor treated as a vector by the label check. 50 picks every
# realistic embedding dim (`abragin/ruBert-style-base` is 768) while still
# skipping incidental short float arrays.
_MIN_RENORMALIZE_SIZE = 50


def _is_centroid_vector(array):
    """True when `array` is a centroid vector rather than metadata.

    One rule, shared by the label check and `load_centroids_npz`, so a small
    float array is metadata in both places (review R2-4).
    """
    return (
        getattr(array, "dtype", None) is not None
        and array.dtype.kind == "f"
        and array.ndim >= 1
        and array.size >= _MIN_RENORMALIZE_SIZE
    )


def _check_npz_label(path, label, expect_encoder, entry_point, vectors=None):
    """Check a centroid npz label, and (when given) the vectors against it.

    `vectors` is the dict of loaded arrays; when given, the file's centroid
    vectors must be exactly the keys of `n_rows_per_group`, and each vector's
    last axis must equal the label's `dim` (reviews F4, R2-4). A key is a
    vector when the label names it, or when it is a float array of at least
    `_MIN_RENORMALIZE_SIZE` elements; a small unnamed float array is metadata
    and is ignored, the same rule `load_centroids_npz` renormalizes by.
    `input_normalization` may be the literal "unknown" for a historical
    centroid whose input norms were never measured; otherwise it must be the
    measured dict that `build_centroids` writes. The historical input counts
    (the values of `n_rows_per_group`) are required as a field but cannot be
    re-derived from the stored means, so they are not re-checked.
    """
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
    require_label_fields(label, CENTROID_LABEL_FIELDS,
                         where=f"{entry_point}: npz {path}", kind="centroid")
    if label["kind"] != "centroid":
        raise StyleLabelError(
            f"{entry_point}: npz {path} label kind is {label['kind']!r}, "
            "expected 'centroid'"
        )
    input_normalization = label["input_normalization"]
    if input_normalization != "unknown" and not isinstance(
        input_normalization, dict
    ):
        raise StyleLabelError(
            f"{entry_point}: npz {path} input_normalization must be a "
            "measured dict or the string 'unknown', "
            f"got {type(input_normalization).__name__}"
        )
    if vectors is not None:
        dim = label["dim"]
        group_keys = set(label["n_rows_per_group"])
        float_keys = {
            key for key, array in vectors.items()
            if getattr(array, "dtype", None) is not None
            and array.dtype.kind == "f" and array.ndim >= 1
        }
        # A file key is a vector when the label names it, or when it is a
        # full-size float array; a small unnamed float array is metadata in
        # both this check and the renormalizer (review R2-4c).
        file_vector_keys = (group_keys & float_keys) | {
            key for key in float_keys if _is_centroid_vector(vectors[key])
        }
        extra = sorted(file_vector_keys - group_keys)
        missing = sorted(group_keys - file_vector_keys)
        if extra or missing:
            raise StyleLabelError(
                f"{entry_point}: npz {path} vector keys do not match "
                f"n_rows_per_group; extra {extra}, missing {missing}"
            )
        for key in file_vector_keys:
            width = vectors[key].shape[-1]
            if width != dim:
                raise StyleLabelError(
                    f"{entry_point}: npz {path} vector {key!r} has width "
                    f"{width}, label says dim {dim}"
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
        expect_encoder = model_names.style_encoder_key()
    file_path = os.path.join(os.path.dirname(__file__), "author_styles.npz")
    with np.load(file_path) as loaded:
        author_styles = {key: loaded[key] for key in loaded
                         if key != "__style_label__"}
    _check_npz_label(file_path, read_style_label(file_path), expect_encoder,
                     "load_author_styles", author_styles)
    return author_styles


AUTHOR_STYLES_PATH = os.path.join(os.path.dirname(__file__),
                                  "author_styles.npz")
# The 5.1-verified base_v1 author centroids, byte-identical to the canonical
# file before the pin move (task 5.2): the supported named-conditioning
# source of the base_v1 TinyStyler checkpoints.
AUTHOR_STYLES_BASE_V1_PATH = os.path.join(os.path.dirname(__file__),
                                          "author_styles_base_v1.npz")


def load_labelled_centroids(path, *, expect_encoder=None):
    """The checked loader: a labelled centroid npz as `LabelledStyleVectors`.

    The file must carry a `__style_label__` that names `expect_encoder`
    (default: the pin, read at call time) with that entry's registry
    hashes, every centroid label field, and vectors matching
    `n_rows_per_group`. An unlabelled file raises, whatever the pin. The
    vectors are returned as stored (no renormalization); the return holds
    private write-protected copies.
    """
    from tst_utils.eval.data.style_store import LabelledStyleVectors
    if expect_encoder is None:
        expect_encoder = model_names.style_encoder_key()
    label = read_style_label(path)
    if label is None:
        raise StyleLabelError(
            f"load_labelled_centroids: unlabelled npz {os.path.realpath(path)}"
        )
    with np.load(path) as loaded:
        vectors = {k: loaded[k] for k in loaded.files if k != "__style_label__"}
    _check_npz_label(path, label, expect_encoder, "load_labelled_centroids",
                     vectors)
    keys = list(label["n_rows_per_group"])
    return LabelledStyleVectors._make(
        keys=keys, vectors=[vectors[key] for key in keys],
        encoder_key=label["encoder_key"], provenance="loader",
        source=os.path.realpath(path), label=label)


def load_labelled_author_styles(expect_encoder=None, *, path=None):
    """The author centroids as `LabelledStyleVectors`, checked on load.

    `path` defaults to the canonical `author_styles.npz`; pass
    `AUTHOR_STYLES_BASE_V1_PATH` with `expect_encoder="base_v1"` for the
    named-conditioning source of a base_v1 checkpoint.
    """
    return load_labelled_centroids(path or AUTHOR_STYLES_PATH,
                                   expect_encoder=expect_encoder)


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
        expect_encoder = model_names.style_encoder_key()
    with np.load(path) as loaded:
        result = {k: loaded[k] for k in loaded.files if k != "__style_label__"}
    _check_npz_label(path, read_style_label(path), expect_encoder,
                     "load_centroids_npz", result)
    if renormalize:
        result = {
            k: (renormalize_centroid(v) if _is_centroid_vector(v) else v)
            for k, v in result.items()
        }
    return result

def load_llm_data():
    """Load TST results performed by LLMs (manually)."""
    file_path = os.path.join(
        os.path.dirname(__file__), "llms_with_scores.csv.gz"
    )
    return pd.read_csv(file_path, compression='gzip')