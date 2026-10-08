"""Labels and guarded IO for stored style embeddings.

A stored style embedding carries its label: the encoder that produced it
and its normalization. Readers check the label against the pin before
they return a vector.
"""

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tst_utils.eval.model_names import STYLE_ENCODER_KEY
from tst_utils.eval.style_encoder_registry import get_encoder


class StyleProvenanceWarning(FutureWarning):
    """A stored style vector is used on provenance assumptions.

    This is a `FutureWarning` and not a `DeprecationWarning`, because
    Python hides a `DeprecationWarning` that fires outside `__main__`
    (PEP 565), and these warnings fire inside `tst_utils`, so a script
    user would not see them. Repeats are removed by the standard warnings
    registry, with no state of our own: the default action shows each
    distinct message once per call site, so each message names its column
    or its npz file and its entry point.
    """


def style_text_key(text):
    """Return the join key of `text`: hex sha256 of its UTF-8 bytes.

    The input is the exact string the encoder receives.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def measure_normalization(embeddings):
    """Measure the normalization of per-row `embeddings`.

    Returns a dict with `normalization` (`"l2"` if every norm is within
    1e-3 of 1, `"none"` otherwise) plus `norm_min` and `norm_max` as
    Python floats. The norms are computed in float64 whatever the input
    dtype.

    Raises:
        ValueError: on an empty input; if some norms are within 1e-3 of 1
            and others are not (a mixed-scale array); or if
            `norm_max / norm_min` exceeds 2.
    """
    norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float64), axis=1)
    if norms.size == 0:
        raise ValueError("measure_normalization needs at least one row")
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    within = np.abs(norms - 1.0) <= 1e-3
    if np.any(within) and not np.all(within):
        raise ValueError(
            "mixed-scale embeddings: some norms are within 1e-3 of 1 "
            f"(min {norm_min}, max {norm_max})"
        )
    if norm_min == 0.0:
        if norm_max > 0.0:
            raise ValueError(
                "mixed-scale embeddings: zero and non-zero norms "
                f"(max {norm_max})"
            )
    elif norm_max / norm_min > 2:
        raise ValueError(
            f"mixed-scale embeddings: norm_max / norm_min exceeds 2 "
            f"(min {norm_min}, max {norm_max})"
        )
    return {
        "normalization": "l2" if bool(np.all(within)) else "none",
        "norm_min": norm_min,
        "norm_max": norm_max,
    }


def warn_provenance(message):
    """Emit a `StyleProvenanceWarning` with stacklevel 2."""
    warnings.warn(message, StyleProvenanceWarning, stacklevel=2)


class StyleLabelError(ValueError):
    """A stored style file fails its label check."""


_FREE_FIELDS = ("run_name", "date", "notes", "provenance")

_ENCODER_LABEL_FIELDS = (
    "encoder_key",
    "file_sha256",
    "dim",
    "n_rows",
    "normalization",
    "norm_min",
    "norm_max",
)


def save_style_embeddings(texts, encoded, path, **free_fields):
    """Save per-row style vectors with their label as a parquet side file.

    `encoded` must be an `EncodedStyle` (checked structurally, so this
    module never imports `eval.metrics.style`). Repeated texts are stored
    once, keeping the first. The label goes into the parquet schema
    metadata under the key `style_label`, as JSON: the encoder label,
    with `n_rows` set to the number of STORED rows, plus the free fields.
    """
    embeddings = getattr(encoded, "embeddings", None)
    encoder_label = getattr(encoded, "label", None)
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.ndim != 2
        or not isinstance(encoder_label, dict)
        or any(key not in encoder_label for key in _ENCODER_LABEL_FIELDS)
    ):
        raise TypeError(
            "encoded must be an EncodedStyle, "
            f"got {type(encoded).__name__}"
        )
    texts = list(texts)
    if len(texts) != encoder_label["n_rows"]:
        raise ValueError(
            f"{len(texts)} texts for {encoder_label['n_rows']} encoded rows"
        )
    for key in free_fields:
        if key not in _FREE_FIELDS:
            raise ValueError(f"unknown free field: {key}")
    seen = set()
    kept = []
    stored_keys = []
    stored_excerpts = []
    for index, text in enumerate(texts):
        key = style_text_key(text)
        if key in seen:
            continue
        seen.add(key)
        kept.append(index)
        stored_keys.append(key)
        stored_excerpts.append(text[:80])
    stacked = np.asarray(embeddings, dtype=np.float16)[kept]
    label = dict(encoder_label)
    label["n_rows"] = len(stored_keys)
    for key, value in free_fields.items():
        if key in label:
            raise ValueError(f"free field overwrites an encoder field: {key}")
        label[key] = value
    schema = pa.schema(
        [
            pa.field("text_key", pa.string()),
            pa.field("text_excerpt", pa.string()),
            pa.field("style_emb", pa.list_(pa.float16(), stacked.shape[1])),
        ],
        metadata={b"style_label": json.dumps(label).encode("utf-8")},
    )
    table = pa.Table.from_arrays(
        [
            pa.array(stored_keys, type=pa.string()),
            pa.array(stored_excerpts, type=pa.string()),
            pa.FixedSizeListArray.from_arrays(
                pa.array(stacked.reshape(-1)), stacked.shape[1]
            ),
        ],
        schema=schema,
    )
    pq.write_table(table, path)


def load_style_embeddings(path, expect_encoder=STYLE_ENCODER_KEY):
    """Load a parquet side file after checking its label.

    The label is read with `pq.read_schema` BEFORE any row, so a file
    whose label mismatches raises without touching the row data.
    """
    schema = pq.read_schema(path)
    raw = (schema.metadata or {}).get(b"style_label")
    if raw is None:
        raise StyleLabelError(f"no style label in {path}")
    label = json.loads(bytes(raw).decode("utf-8"))
    if label.get("encoder_key") != expect_encoder:
        raise StyleLabelError(
            f"style file {path} names encoder {label.get('encoder_key')!r}, "
            f"expected {expect_encoder!r}"
        )
    if label.get("file_sha256") != get_encoder(expect_encoder)["file_sha256"]:
        raise StyleLabelError(
            f"style file {path} names a file_sha256 that differs from "
            f"the registry entry of {expect_encoder!r}"
        )
    return pq.read_table(path).to_pandas(), label


def join_style_embeddings(df, path, text_col, out_col,
                          expect_encoder=STYLE_ENCODER_KEY):
    """Join a side file onto `df` by exact-text key.

    Returns a copy; `df` is not mutated and nothing is written to
    `df.attrs`. `out_col` holds float32 1D np.ndarray per row, the same
    form as the inline columns.
    """
    if out_col in df.columns:
        raise ValueError(f"output column already exists: {out_col}")
    frame, _label = load_style_embeddings(path, expect_encoder)
    key_to_row = {key: row for row, key in enumerate(frame["text_key"])}
    stacked = (
        np.stack(frame["style_emb"].to_numpy()).astype(np.float32)
        if len(frame)
        else np.zeros((0, 0), dtype=np.float32)
    )
    keys = [style_text_key(text) for text in df[text_col]]
    missing = [
        str(text)[:80] for key, text in zip(keys, df[text_col])
        if key not in key_to_row
    ]
    if missing:
        raise StyleLabelError(
            f"{len(missing)} texts have no stored row, "
            f"first missing excerpt: {missing[0]!r}"
        )
    joined = df.copy()
    joined[out_col] = [stacked[key_to_row[key]] for key in keys]
    return joined


def _require_encoded_style(encoded):
    embeddings = getattr(encoded, "embeddings", None)
    encoder_label = getattr(encoded, "label", None)
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.ndim != 2
        or not isinstance(encoder_label, dict)
        or any(key not in encoder_label for key in _ENCODER_LABEL_FIELDS)
    ):
        raise TypeError(
            "encoded must be an EncodedStyle, "
            f"got {type(encoded).__name__}"
        )
    return np.asarray(embeddings), encoder_label


def build_centroids(encoded, groups, *, renormalize, path, **free_fields):
    """Build one centroid per group from fresh per-row vectors.

    `encoded` must be an `EncodedStyle`, so the encoder key comes from the
    encode call. `groups` holds one group name per row. The means are
    computed in float64; with `renormalize` each mean is L2-normalized.
    The centroids are written as fp16 with `np.savez`, one entry per
    group, plus `__style_label__` (a JSON string). There is no writer
    that accepts finished centroids with a free label.

    Returns:
        (dict, dict): the centroids (`{group: fp16 array}`) and the label.
    """
    embeddings, encoder_label = _require_encoded_style(encoded)
    groups = list(groups)
    if len(groups) != embeddings.shape[0]:
        raise ValueError(
            f"{len(groups)} groups for {embeddings.shape[0]} encoded rows"
        )
    if "__style_label__" in groups:
        raise ValueError("a group may not be named __style_label__")
    input_measurement = measure_normalization(embeddings)
    in_float64 = np.asarray(embeddings, dtype=np.float64)
    centroids = {}
    n_rows_per_group = {}
    order = []
    for index, group in enumerate(groups):
        if group not in n_rows_per_group:
            n_rows_per_group[group] = []
            order.append(group)
        n_rows_per_group[group].append(index)
    for group in order:
        mean = in_float64[n_rows_per_group[group]].mean(axis=0)
        if renormalize:
            mean = mean / np.linalg.norm(mean)
        centroids[group] = mean.astype(np.float16)
    norms = np.linalg.norm(
        np.stack([centroids[group] for group in order]).astype(np.float64),
        axis=1,
    )
    label = {
        "kind": "centroid",
        "encoder_key": encoder_label["encoder_key"],
        "file_sha256": dict(encoder_label["file_sha256"]),
        "dim": int(centroids[order[0]].shape[0]),
        "input_normalization": input_measurement,
        "renormalized": bool(renormalize),
        "n_rows_per_group": {
            group: len(n_rows_per_group[group]) for group in order
        },
        "centroid_norm_min": float(np.min(norms)),
        "centroid_norm_max": float(np.max(norms)),
    }
    for key in free_fields:
        if key not in _FREE_FIELDS:
            raise ValueError(f"unknown free field: {key}")
        if key in label:
            raise ValueError(f"free field overwrites a label field: {key}")
    label.update(free_fields)
    np.savez(path, **centroids, __style_label__=json.dumps(label))
    return centroids, label
