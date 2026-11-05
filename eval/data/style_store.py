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

from tst_utils.eval import model_names
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

# The required label schema, in one place, mirroring exactly what the writers
# write. `encode_style` -> `save_style_embeddings` writes the per-row side-file
# label; `build_centroids` writes the centroid npz label. The readers require
# these fields, so a truncated or hand-built label cannot pass as checked
# (review F4). `input_normalization` may be the literal "unknown" for a
# historical centroid whose input norms were never measured.
SIDE_LABEL_FIELDS = (
    "encoder_key",
    "file_sha256",
    "dim",
    "n_rows",
    "normalization",
    "norm_min",
    "norm_max",
)

CENTROID_LABEL_FIELDS = (
    "kind",
    "encoder_key",
    "file_sha256",
    "dim",
    "input_normalization",
    "renormalized",
    "n_rows_per_group",
    "centroid_norm_min",
    "centroid_norm_max",
)


def _missing_label_fields(label, required):
    """Return the required field names absent from `label`."""
    return [field for field in required if field not in label]


def require_label_fields(label, required, *, where, kind):
    """Raise `StyleLabelError` if `label` lacks any required field."""
    missing = _missing_label_fields(label, required)
    if missing:
        raise StyleLabelError(
            f"{where}: {kind} label is missing required field(s): {missing}"
        )


def _check_text_keys(encoded, texts, entry_point):
    """Raise `StyleLabelError` unless `texts` matches `encoded.text_keys`.

    The text/vector association is enforced, not assumed (review F3): every
    supplied text's `style_text_key` must equal the retained
    ``encoded.text_keys`` at the same position. The writer never reorders
    silently, so a changed order or a text edited after the encode raises,
    naming the first differing position. A hand-built `EncodedStyle` without
    retained keys is rejected, because the association cannot be checked.
    """
    retained_keys = tuple(getattr(encoded, "text_keys", ()) or ())
    if not retained_keys:
        raise StyleLabelError(
            f"{entry_point}: encoded carries no text keys (it did "
            "not come from encode_style), so the text/vector association "
            "cannot be checked"
        )
    supplied_keys = [style_text_key(text) for text in texts]
    if list(retained_keys) != supplied_keys:
        position = min(len(retained_keys), len(supplied_keys))
        for index, (retained, supplied) in enumerate(
            zip(retained_keys, supplied_keys)
        ):
            if retained != supplied:
                position = index
                break
        raise StyleLabelError(
            f"{entry_point}: the supplied text at position "
            f"{position} differs from the encoded input (the order changed, "
            "or the text was edited after the encode)"
        )


def save_style_embeddings(texts, encoded, path, **free_fields):
    """Save per-row style vectors with their label as a parquet side file.

    `encoded` must be an `EncodedStyle` (checked structurally, so this
    module never imports `eval.metrics.style`). Repeated texts are stored
    once, keeping the first. The label goes into the parquet schema
    metadata under the key `style_label`, as JSON: the encoder label,
    with `n_rows` set to the number of STORED rows, plus the free fields.

    The text/vector association is enforced, not assumed (review F3):
    every supplied text's ``style_text_key`` must equal the retained
    ``encoded.text_keys`` at the same position. The writer never reorders
    silently, so a changed order or a text edited after the encode raises
    `StyleLabelError` naming the first differing position. `encoded` is
    expected to come from `encode_style`; a hand-built `EncodedStyle`
    without retained keys is rejected, because the association cannot be
    checked.
    """
    embeddings = getattr(encoded, "embeddings", None)
    encoder_label = getattr(encoded, "label", None)
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.ndim != 2
        or not isinstance(encoder_label, dict)
        or any(key not in encoder_label for key in SIDE_LABEL_FIELDS)
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
    _check_text_keys(encoded, texts, "save_style_embeddings")
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


def load_style_embeddings(path, expect_encoder=None):
    """Load a parquet side file after checking its label.

    The label is read with `pq.read_schema` BEFORE any row, so a file
    whose label mismatches raises without touching the row data. The
    label must carry every field of `SIDE_LABEL_FIELDS` (review F4), and
    the label is then checked against the file itself: `n_rows` equals
    the stored row count and `dim` equals the `style_emb` vector width.
    `expect_encoder=None` means the pin, read at call time.
    """
    if expect_encoder is None:
        expect_encoder = model_names.style_encoder_key()
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
    require_label_fields(label, SIDE_LABEL_FIELDS,
                          where=f"style file {path}", kind="side-file")
    table = pq.read_table(path)
    n_rows = len(table)
    if int(label["n_rows"]) != n_rows:
        raise StyleLabelError(
            f"style file {path} label says n_rows {label['n_rows']} but the "
            f"file has {n_rows} rows"
        )
    width = getattr(table.schema.field("style_emb").type, "list_size", None)
    if width is not None and int(label["dim"]) != int(width):
        raise StyleLabelError(
            f"style file {path} label says dim {label['dim']} but style_emb "
            f"has width {width}"
        )
    return table.to_pandas(), label


def join_style_embeddings(df, path, text_col, out_col,
                          expect_encoder=None):
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


def guard_inline_style_columns(df, *, source_style_path=None, entry_point):
    """Drop or join unlabelled inline style columns at a pipeline entry.

    `entry_point` (required keyword) names the caller in warning texts.
    Returns a new frame; `df` is not mutated.
    """
    frame = df.copy()
    if source_style_path is not None:
        if "text_style_emb" in frame.columns:
            frame = frame.drop(columns=["text_style_emb"])
        frame = join_style_embeddings(frame, source_style_path, "text",
                                      "text_style_emb")
    elif "text_style_emb" in frame.columns:
        warn_provenance(
            f"{entry_point}: stored column text_style_emb has no label; "
            "dropped, it is encoded fresh"
        )
        frame = frame.drop(columns=["text_style_emb"])
    if "target_style_emb" in frame.columns:
        if model_names.style_encoder_key() == "base_v1":
            warn_provenance(
                f"{entry_point}: stored column target_style_emb has no "
                "label; an unlabelled file predates the label"
            )
        else:
            raise StyleLabelError(
                f"{entry_point}: stored column target_style_emb has no "
                f"label, and the pin is {model_names.style_encoder_key()!r}"
            )
    return frame


def _require_encoded_style(encoded):
    embeddings = getattr(encoded, "embeddings", None)
    encoder_label = getattr(encoded, "label", None)
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.ndim != 2
        or not isinstance(encoder_label, dict)
        or any(key not in encoder_label for key in SIDE_LABEL_FIELDS)
    ):
        raise TypeError(
            "encoded must be an EncodedStyle, "
            f"got {type(encoded).__name__}"
        )
    return np.asarray(embeddings), encoder_label


def build_centroids(encoded, groups, *, renormalize, path, texts=None,
                    **free_fields):
    """Build one centroid per group from fresh per-row vectors.

    `encoded` must be an `EncodedStyle`, so the encoder key comes from the
    encode call. `groups` holds one group name per row, or a tuple of group
    names for a row that belongs to several groups (a pooled group, such as
    the step-4 `web_informal`, task 5.2). The means are
    computed in float64; with `renormalize` each mean is L2-normalized.
    The centroids are written as fp16 with `np.savez`, one entry per
    group, plus `__style_label__` (a JSON string). There is no writer
    that accepts finished centroids with a free label.

    `texts`, when given, are the texts the encode saw, in row order; they
    are checked against `encoded.text_keys` by the same rule the side-file
    writer uses (review R2-5), so a reordered or filtered list cannot write
    the wrong centroid under a correct label.

    Returns:
        (dict, dict): the centroids (`{group: fp16 array}`) and the label.
    """
    embeddings, encoder_label = _require_encoded_style(encoded)
    groups = list(groups)
    if len(groups) != embeddings.shape[0]:
        raise ValueError(
            f"{len(groups)} groups for {embeddings.shape[0]} encoded rows"
        )
    memberships = [
        tuple(group) if isinstance(group, (tuple, list)) else (group,)
        for group in groups
    ]
    if any(not names for names in memberships):
        raise ValueError("every row needs at least one group")
    if any(len(set(names)) != len(names) for names in memberships):
        raise ValueError("a row names the same group twice")
    if any("__style_label__" in names for names in memberships):
        raise ValueError("a group may not be named __style_label__")
    if texts is not None:
        _check_text_keys(encoded, texts, "build_centroids")
    input_measurement = measure_normalization(embeddings)
    in_float64 = np.asarray(embeddings, dtype=np.float64)
    centroids = {}
    n_rows_per_group = {}
    order = []
    for index, names in enumerate(memberships):
        for group in names:
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


# ---------------- task 5.2: labelled style vectors (requirement 4) ----------

PROVENANCES = ("loader", "build", "encode", "declare")


def _frozen_matrix(vectors, keys):
    """A private, write-protected float copy of `vectors`, one row per key."""
    rows = [np.array(vector, copy=True) for vector in vectors]
    if not rows:
        raise ValueError("labelled style vectors need at least one vector")
    matrix = np.stack(rows)
    if matrix.ndim != 2:
        raise ValueError(
            f"each style vector must be 1D, got shape {matrix.shape[1:]}"
        )
    if matrix.dtype.kind != "f":
        raise ValueError(f"style vectors must be float, got {matrix.dtype}")
    if len(keys) != matrix.shape[0]:
        raise ValueError(f"{len(keys)} keys for {matrix.shape[0]} vectors")
    matrix.setflags(write=False)
    return matrix


def _freeze_label(label):
    """A deep JSON copy of `label`, so a caller's dict cannot change it."""
    return json.loads(json.dumps(label))


class LabelledStyleVectors:
    """Style vectors bound to their encoder label by ownership (task 5.2).

    One vector per key. The object holds a private write-protected copy of
    every vector, an immutable tuple of unique keys and a frozen label, so
    nothing a caller does to its own arrays, dicts or key lists after
    construction changes what the pipeline reads. It is built only by:

    - the checked loader `load_labelled_centroids` (provenance "loader");
    - `from_build`, from the return value of `build_centroids` ("build");
    - `from_encoded`, an explicit conversion of `encode_style` output
      ("encode"); the caller gives one key per encoded row;
    - `declare`, an operator assertion that `vectors` came from
      `encoder_key` ("declare"); it is reported as a declaration.

    Values are stored as supplied: no normalization. A consumer that needs
    a unit view derives it from `vector`/`matrix` and never writes it back.
    The guard catches accidental mismatch, not a caller who builds or
    declares a wrong object on purpose.
    """

    __slots__ = ("_keys", "_index", "_matrix", "_encoder_key", "_provenance",
                 "_source", "_label")

    def __init__(self, *, keys, vectors, encoder_key, provenance, source,
                 label, _token=None):
        if _token is not _CONSTRUCT:
            raise TypeError(
                "build LabelledStyleVectors through load_labelled_centroids, "
                "from_build, from_encoded or declare"
            )
        if provenance not in PROVENANCES:
            raise ValueError(f"unknown provenance: {provenance!r}")
        get_encoder(encoder_key)  # an unregistered key raises here
        keys = tuple(keys)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("style vector keys must be strings")
        if len(set(keys)) != len(keys):
            raise ValueError("style vector keys must be unique")
        if any(key.startswith("__") for key in keys):
            raise ValueError("a style vector key may not start with '__'")
        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_index",
                           {key: row for row, key in enumerate(keys)})
        object.__setattr__(self, "_matrix", _frozen_matrix(vectors, keys))
        object.__setattr__(self, "_encoder_key", encoder_key)
        object.__setattr__(self, "_provenance", provenance)
        object.__setattr__(self, "_source", str(source))
        object.__setattr__(self, "_label", _freeze_label(label))

    def __setattr__(self, name, value):
        raise AttributeError("LabelledStyleVectors is immutable")

    def __delattr__(self, name):
        raise AttributeError("LabelledStyleVectors is immutable")

    # -- constructors -------------------------------------------------------

    @classmethod
    def _make(cls, **kwargs):
        return cls(_token=_CONSTRUCT, **kwargs)

    @classmethod
    def from_build(cls, built, *, source):
        """From the `(centroids, label)` pair that `build_centroids` returns."""
        centroids, label = built
        keys = list(centroids)
        return cls._make(keys=keys, vectors=[centroids[k] for k in keys],
                         encoder_key=label["encoder_key"], provenance="build",
                         source=source, label=label)

    @classmethod
    def from_encoded(cls, encoded, *, keys, source="encode_style"):
        """An explicit conversion of `encode_style` output; `keys` is required.

        `keys` gives one caller key per encoded row, in row order. The
        encoder key comes from the encode call's label.
        """
        from tst_utils.eval.metrics.style import EncodedStyle
        if not isinstance(encoded, EncodedStyle):
            raise TypeError(
                f"from_encoded needs an EncodedStyle, got "
                f"{type(encoded).__name__}"
            )
        embeddings, label = _require_encoded_style(encoded)
        keys = list(keys)
        return cls._make(keys=keys, vectors=list(embeddings),
                         encoder_key=label["encoder_key"], provenance="encode",
                         source=source, label=label)

    @classmethod
    def declare(cls, vectors, encoder_key, source, *, keys):
        """An operator assertion: `vectors` came from `encoder_key`.

        `vectors` is a sequence (or 2D array) in the order of `keys`, or a
        dict whose iteration order matches `keys`. Reported as a
        declaration by `describe()` and in `provenance`.
        """
        keys = list(keys)
        if isinstance(vectors, dict):
            if list(vectors) != keys:
                raise ValueError("declare: the dict keys must equal keys, "
                                 "in the same order")
            vectors = [vectors[key] for key in keys]
        rows = [np.asarray(vector) for vector in vectors]
        label = {"encoder_key": encoder_key, "declared": True,
                 "source": str(source)}
        return cls._make(keys=keys, vectors=rows, encoder_key=encoder_key,
                         provenance="declare", source=source, label=label)

    # -- reading ------------------------------------------------------------

    @property
    def keys(self):
        return self._keys

    @property
    def encoder_key(self):
        return self._encoder_key

    @property
    def provenance(self):
        return self._provenance

    @property
    def source(self):
        return self._source

    @property
    def is_declared(self):
        return self._provenance == "declare"

    @property
    def label(self):
        """A copy of the frozen label."""
        return _freeze_label(self._label)

    @property
    def dim(self):
        return int(self._matrix.shape[1])

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._index

    def vector(self, key):
        """The stored vector of `key`: a read-only view of the private copy."""
        if key not in self._index:
            raise KeyError(f"unknown style key {key!r}; known: "
                           f"{list(self._keys)}")
        return self._matrix[self._index[key]]

    def matrix(self):
        """All stored vectors, one row per key, read-only."""
        return self._matrix

    def norms(self):
        return np.linalg.norm(np.asarray(self._matrix, dtype=np.float64),
                              axis=1)

    def describe(self):
        """A short record for logs and persisted results."""
        record = {"encoder_key": self._encoder_key,
                  "provenance": self._provenance, "source": self._source,
                  "n_keys": len(self._keys)}
        if self.is_declared:
            record["declared"] = True
        return record

    def select(self, row_keys):
        """A selection: this source and an ordered sequence of row keys."""
        return StyleSelection(self, row_keys)

    def __repr__(self):
        return (f"LabelledStyleVectors({self._encoder_key!r}, "
                f"{self._provenance!r}, keys={list(self._keys)!r})")


_CONSTRUCT = object()


class StyleSelection:
    """A labelled source with an ordered, immutable sequence of row keys.

    Keys may repeat and come in any order; each must exist in the source.
    The keys are copied into a tuple at validation, so a later change to
    the caller's list changes nothing. `vectors()` gathers the rows from the
    source's private copies with no new label, no deduplication and no
    reordering.
    """

    __slots__ = ("_source", "_row_keys")

    def __init__(self, source, row_keys):
        if not isinstance(source, LabelledStyleVectors):
            raise TypeError(
                f"a selection needs LabelledStyleVectors, got "
                f"{type(source).__name__}"
            )
        if isinstance(row_keys, str):
            raise TypeError("row_keys must be a sequence of keys, not a str")
        row_keys = tuple(row_keys)
        unknown = sorted({key for key in row_keys if key not in source})
        if unknown:
            raise KeyError(f"unknown style keys {unknown}; known: "
                           f"{list(source.keys)}")
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_row_keys", row_keys)

    def __setattr__(self, name, value):
        raise AttributeError("StyleSelection is immutable")

    @property
    def source(self):
        return self._source

    @property
    def row_keys(self):
        return self._row_keys

    @property
    def encoder_key(self):
        return self._source.encoder_key

    def __len__(self):
        return len(self._row_keys)

    def vectors(self):
        """One row per row key, gathered from the source; a fresh array."""
        return np.stack([self._source.vector(key) for key in self._row_keys])
