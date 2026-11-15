"""Labels and guarded IO for stored style embeddings.

A stored style embedding carries its label: the encoder that produced it
and its normalization. Readers check the label against the pin before
they return a vector.
"""

import glob
import hashlib
import json
import logging
import os
import shutil
import tempfile
import warnings
import weakref

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (pa.compute)
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


_NORM_CHUNK_ROWS = 65536


def _row_norms_float64(embeddings, chunk_rows=None):
    """Return the float64 L2 norm of each row of `embeddings`.

    The rows are converted to float64 one chunk at a time, so the extra
    memory is one chunk (about 0.8 GB at 768 dims), not a float64 copy of
    the whole array plus its squares (about 4 times the size of an fp32
    input; about 17 GiB at 1.46 million rows of 768 dims). Each row's norm
    is the same computation as on the whole array, so the result does not
    depend on `chunk_rows`.
    """
    if chunk_rows is None:
        chunk_rows = _NORM_CHUNK_ROWS
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] <= chunk_rows:
        return np.linalg.norm(
            np.asarray(embeddings, dtype=np.float64), axis=1
        )
    norms = np.empty(embeddings.shape[0], dtype=np.float64)
    for start in range(0, embeddings.shape[0], chunk_rows):
        stop = start + chunk_rows
        norms[start:stop] = np.linalg.norm(
            np.asarray(embeddings[start:stop], dtype=np.float64), axis=1
        )
    return norms


def measure_normalization(embeddings):
    """Measure the normalization of per-row `embeddings`.

    Returns a dict with `normalization` (`"l2"` if every norm is within
    1e-3 of 1, `"none"` otherwise) plus `norm_min` and `norm_max` as
    Python floats. The norms are computed in float64 whatever the input
    dtype, in chunks of rows (`_row_norms_float64`).

    Raises:
        ValueError: on an empty input; if some norms are within 1e-3 of 1
            and others are not (a mixed-scale array); or if
            `norm_max / norm_min` exceeds 2.
    """
    norms = _row_norms_float64(embeddings)
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
    _write_side_table(path, stored_keys, stored_excerpts, stacked, label)


def _write_side_table(path, stored_keys, stored_excerpts, stacked, label):
    """Write the side-file table: keys, excerpts, fp16 rows, the label."""
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


def _read_side_label(path):
    """The label of a side file, read from the schema only (no row)."""
    schema = pq.read_schema(path)
    raw = (schema.metadata or {}).get(b"style_label")
    if raw is None:
        raise StyleLabelError(f"no style label in {path}")
    return json.loads(bytes(raw).decode("utf-8"))


def _check_side_label(label, expect_encoder, where):
    """The label checks of the loader: the encoder against `expect_encoder`
    (None means the pin, read at call time), `file_sha256` against the
    registry entry, and the required fields."""
    if expect_encoder is None:
        expect_encoder = model_names.style_encoder_key()
    if label.get("encoder_key") != expect_encoder:
        raise StyleLabelError(
            f"{where} names encoder {label.get('encoder_key')!r}, "
            f"expected {expect_encoder!r}"
        )
    if label.get("file_sha256") != get_encoder(expect_encoder)["file_sha256"]:
        raise StyleLabelError(
            f"{where} names a file_sha256 that differs from "
            f"the registry entry of {expect_encoder!r}"
        )
    require_label_fields(label, SIDE_LABEL_FIELDS, where=where,
                         kind="side-file")


SIDE_PART_GLOB = "part-*.parquet"

# The label fields that every part of a side folder must share.
_SHARED_PART_FIELDS = ("encoder_key", "file_sha256", "dim", "normalization")


def side_part_paths(folder):
    """The part files of a side folder, sorted; raises if there is none."""
    parts = sorted(glob.glob(os.path.join(folder, SIDE_PART_GLOB)))
    if not parts:
        raise StyleLabelError(f"style folder {folder} has no {SIDE_PART_GLOB}")
    return parts


def _read_side_table(path, expect_encoder):
    """The checked table and label of one side file (see the loader)."""
    label = _read_side_label(path)
    _check_side_label(label, expect_encoder, f"style file {path}")
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
    return table, label


def _combine_part_labels(folder, labels, part_paths, n_rows):
    """The label of a side folder: the shared fields, the total row count,
    the norm range over the parts, and each part's own label in `parts`."""
    first = labels[0]
    for path, label in zip(part_paths[1:], labels[1:]):
        for field in _SHARED_PART_FIELDS:
            if label.get(field) != first.get(field):
                raise StyleLabelError(
                    f"style folder {folder}: part {os.path.basename(path)} "
                    f"differs in {field}: {label.get(field)!r} != "
                    f"{first.get(field)!r}")
    norm_min = min(float(label["norm_min"]) for label in labels)
    norm_max = max(float(label["norm_max"]) for label in labels)
    if norm_min > 0 and norm_max / norm_min > 2:
        raise StyleLabelError(
            f"style folder {folder}: mixed-scale parts, norm_max / norm_min "
            f"exceeds 2 (min {norm_min}, max {norm_max})")
    combined = {field: first[field] for field in _SHARED_PART_FIELDS}
    combined.update({"n_rows": n_rows, "norm_min": norm_min,
                     "norm_max": norm_max})
    combined["parts"] = [
        dict(label, file=os.path.basename(path))
        for path, label in zip(part_paths, labels)]
    return combined


def load_style_embeddings(path, expect_encoder=None):
    """Load a parquet side file, or a side folder, after checking its label.

    The label is read with `pq.read_schema` BEFORE any row, so a file
    whose label mismatches raises without touching the row data. The
    label must carry every field of `SIDE_LABEL_FIELDS` (review F4), and
    the label is then checked against the file itself: `n_rows` equals
    the stored row count and `dim` equals the `style_emb` vector width.
    `expect_encoder=None` means the pin, read at call time.

    A side folder holds part files `part-*.parquet`, each a side file
    with its own label. Every part passes the checks above; the parts
    must agree in `encoder_key`, `file_sha256`, `dim` and
    `normalization`, hold no text key twice, and have a norm range of at
    most 2. The frame is the parts in file-name order; the label holds
    the shared fields, the total `n_rows`, the norm range, and each
    part's label (with its file name) in `parts`.
    """
    if not os.path.isdir(path):
        table, label = _read_side_table(path, expect_encoder)
        return table.to_pandas(), label
    part_paths = side_part_paths(path)
    part_labels = []
    for part in part_paths:  # every label before any row
        part_label = _read_side_label(part)
        _check_side_label(part_label, expect_encoder, f"style file {part}")
        part_labels.append(part_label)
    _combine_part_labels(path, part_labels, part_paths, 0)
    tables, labels = [], []
    for part in part_paths:
        table, label = _read_side_table(part, expect_encoder)
        tables.append(table)
        labels.append(label)
    table = pa.concat_tables(tables)
    del tables
    keys = table.column("text_key")
    if pa.compute.count_distinct(keys).as_py() != len(table):
        raise StyleLabelError(
            f"style folder {path}: a text key is stored in more than one part")
    label = _combine_part_labels(path, labels, part_paths, len(table))
    return table.to_pandas(), label


_logger = logging.getLogger(__name__)

# The key -> row index of a preloaded frame, built once per frame object.
# Keyed by id(); the weak reference proves the id still names that frame.
_PRELOADED_INDEX = {}


def _preloaded_index(frame):
    entry = _PRELOADED_INDEX.get(id(frame))
    if entry is not None and entry[0]() is frame and entry[1] == len(frame):
        return entry[2]
    index = {key: row for row, key in enumerate(frame["text_key"])}
    frame_id = id(frame)
    _PRELOADED_INDEX[frame_id] = (
        weakref.ref(frame, lambda _ref: _PRELOADED_INDEX.pop(frame_id, None)),
        len(frame), index)
    return index


def _read_side_rows(path, expect_encoder, wanted_keys):
    """The label of a side file or folder, and the stored rows of
    `wanted_keys` only: (label, rows, row_of).

    The checks are those of `load_style_embeddings`: every label before
    any row, `n_rows` and `dim` against each file, the shared fields and
    the norm range of a folder, no key in two parts. Then only the
    `text_key` column of each file is read; the `style_emb` column is read
    one file at a time and only the matching rows are kept. `rows` is an
    (n_found, dim) array of the stored dtype; `row_of` maps each found key
    to its row in `rows`. Memory grows with the wanted rows, not the file.
    """
    is_folder = os.path.isdir(path)
    part_paths = side_part_paths(path) if is_folder else [path]
    labels = []
    for part in part_paths:  # every label before any row
        part_label = _read_side_label(part)
        _check_side_label(part_label, expect_encoder, f"style file {part}")
        labels.append(part_label)
    if is_folder:
        _combine_part_labels(path, labels, part_paths, 0)
    key_columns = []
    for part, part_label in zip(part_paths, labels):
        parquet = pq.ParquetFile(part)
        n_rows = parquet.metadata.num_rows
        if int(part_label["n_rows"]) != n_rows:
            raise StyleLabelError(
                f"style file {part} label says n_rows {part_label['n_rows']} "
                f"but the file has {n_rows} rows")
        width = getattr(parquet.schema_arrow.field("style_emb").type,
                        "list_size", None)
        if width is not None and int(part_label["dim"]) != int(width):
            raise StyleLabelError(
                f"style file {part} label says dim {part_label['dim']} but "
                f"style_emb has width {width}")
        key_columns.append(
            pq.read_table(part, columns=["text_key"]).column("text_key"))
    if is_folder:
        n_total = sum(len(keys) for keys in key_columns)
        all_keys = pa.chunked_array(
            [chunk for keys in key_columns for chunk in keys.chunks],
            type=pa.string())
        if pa.compute.count_distinct(all_keys).as_py() != n_total:
            raise StyleLabelError(
                f"style folder {path}: a text key is stored in more than "
                f"one part")
        del all_keys
        label = _combine_part_labels(path, labels, part_paths, n_total)
    else:
        label = labels[0]
    dim = int(label["dim"])
    wanted = pa.array(list(wanted_keys), type=pa.string())
    found_keys, found_rows = [], []
    for part, keys in zip(part_paths, key_columns):
        mask = pa.compute.is_in(keys, value_set=wanted)
        picks = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        if not len(picks):
            continue
        picks = pa.array(picks)
        emb = pq.read_table(part, columns=["style_emb"]).column("style_emb")
        taken = emb.take(picks).combine_chunks()
        del emb
        found_rows.append(
            taken.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim))
        found_keys.extend(keys.take(picks).to_pylist())
    rows = (np.concatenate(found_rows) if found_rows
            else np.zeros((0, dim), dtype=np.float16))
    return label, rows, {key: row for row, key in enumerate(found_keys)}


def _side_rows(path, expect_encoder, wanted_keys):
    """(label, rows, row_of, name) of the stored rows of `wanted_keys`,
    for a path or a preloaded pair."""
    if isinstance(path, tuple):
        if len(path) != 2 or not isinstance(path[0], pd.DataFrame) \
                or not isinstance(path[1], dict):
            raise TypeError(
                "path must be a side-file path or the (frame, label) pair "
                "that load_style_embeddings returns")
        frame, label = path
        where = f"preloaded side file (run {label.get('run_name')!r})"
        _check_side_label(label, expect_encoder, where)
        if int(label["n_rows"]) != len(frame):
            raise StyleLabelError(
                f"{where}: label says n_rows {label['n_rows']} but the frame "
                f"has {len(frame)} rows")
        index = _preloaded_index(frame)
        row_of, picks = {}, []
        for key in wanted_keys:
            row = index.get(key)
            if row is not None:
                row_of[key] = len(picks)
                picks.append(row)
        dim = int(label["dim"])
        stored = frame["style_emb"].to_numpy()
        rows = (np.stack([stored[row] for row in picks]) if picks
                else np.zeros((0, dim), dtype=np.float16))
        return label, rows, row_of, where
    label, rows, row_of = _read_side_rows(path, expect_encoder, wanted_keys)
    return label, rows, row_of, str(path)


def _encode_for_label(texts, label, encode_fn, where, entry_point):
    """Encode `texts` with the encoder and normalization of `label`.

    The encoder key and the normalization come from the label, never from
    the caller. The fresh vectors must carry the text keys of `texts`, one
    row per text, and a label equal to `label` in `encoder_key`,
    `file_sha256`, `dim` and `normalization`; otherwise `StyleLabelError`
    names the field and nothing is returned.
    """
    encoded = encode_fn(texts, label["encoder_key"],
                        normalize=(label["normalization"] == "l2"))
    embeddings, fresh_label = _require_encoded_style(encoded)
    if embeddings.shape[0] != len(texts):
        raise StyleLabelError(
            f"{entry_point}: encode_fn returned {embeddings.shape[0]} rows "
            f"for {len(texts)} texts")
    _check_text_keys(encoded, texts, entry_point)
    for field in _SHARED_PART_FIELDS:
        if fresh_label.get(field) != label.get(field):
            raise StyleLabelError(
                f"{entry_point}: the fresh vectors differ from {where} in "
                f"{field}: {fresh_label.get(field)!r} != {label.get(field)!r}")
    return embeddings, encoded


def join_style_embeddings(df, path, text_col, out_col,
                          expect_encoder=None, *, on_missing="raise",
                          encode_fn=None, max_missing_fraction=0.01):
    """Join a side file onto `df` by exact-text key.

    Returns a copy; `df` is not mutated and nothing is written to
    `df.attrs`. `out_col` holds float32 1D np.ndarray per row, the same
    form as the inline columns. Only the stored rows that `df` needs are
    gathered and cast to float32, so a joined column holds no view of the
    whole file.

    `path` is a side-file path (a file or a folder of parts), or the
    `(frame, label)` pair that `load_style_embeddings` returns, unmodified.
    A path gets the checks of `load_style_embeddings`, but only the
    `text_key` column is read whole; the vectors are read one file at a
    time and only the rows `df` needs are kept, so the memory of the join
    grows with `df`, not with the file. The pair is checked on every call
    (`expect_encoder`, the registry hashes, `n_rows` against the frame),
    and its key index is built once per frame object.

    `on_missing="raise"` (the default): a text with no stored row raises
    `StyleLabelError`. `on_missing="encode"` needs `encode_fn` (the
    convention of `encode_style`): if the unique missing texts are more
    than `max_missing_fraction` of the unique texts of the column, it
    raises `StyleLabelError`; otherwise it encodes them with the encoder
    and normalization of the label, rounds the fresh vectors through fp16
    (so a text has the same vector stored or fresh), and logs one line
    with the path, the missing count and the total. The file is never
    written: persist with `extend_style_embeddings`.
    """
    if out_col in df.columns:
        raise ValueError(f"output column already exists: {out_col}")
    if on_missing not in ("raise", "encode"):
        raise ValueError(f"on_missing must be 'raise' or 'encode', "
                         f"got {on_missing!r}")
    if on_missing == "encode" and encode_fn is None:
        raise TypeError("on_missing='encode' requires encode_fn")
    texts = list(df[text_col])
    keys = [style_text_key(text) for text in texts]
    label, stored_rows, row_of, where = _side_rows(
        path, expect_encoder, dict.fromkeys(keys))
    missing = {}
    for key, text in zip(keys, texts):
        if key not in row_of and key not in missing:
            missing[key] = text
    if missing and on_missing == "raise":
        first = next(iter(missing.values()))
        n_missing_rows = sum(1 for key in keys if key in missing)
        raise StyleLabelError(
            f"{n_missing_rows} texts have no stored row, "
            f"first missing excerpt: {str(first)[:80]!r}"
        )
    n_found = len(row_of)
    n_unique = n_found + len(missing)
    if missing:
        fraction = len(missing) / n_unique
        if fraction > max_missing_fraction:
            raise StyleLabelError(
                f"{where}: {len(missing)} of {n_unique} unique texts of "
                f"{text_col!r} have no stored row ({fraction:.4f} > "
                f"max_missing_fraction {max_missing_fraction})")
    dim = int(label["dim"])
    gathered = np.zeros((n_unique, dim), dtype=np.float32)
    gathered[:n_found] = stored_rows
    del stored_rows
    position = dict(row_of)
    if missing:
        missing_texts = list(missing.values())
        fresh, _ = _encode_for_label(missing_texts, label, encode_fn, where,
                                     "join_style_embeddings")
        gathered[n_found:] = np.asarray(fresh).astype(
            np.float16).astype(np.float32)
        for offset, key in enumerate(missing):
            position[key] = n_found + offset
        _logger.warning(
            "join_style_embeddings: %s: encoded %d missing of %d unique "
            "texts of %r", where, len(missing), n_unique, text_col)
    joined = df.copy()
    joined[out_col] = [gathered[position[key]] for key in keys]
    return joined


def extend_style_embeddings(texts, path, *, out_path, encode_fn,
                            **free_fields):
    """Add the vectors of the texts that `path` lacks; write to `out_path`.

    `out_path` is required; `out_path == path` is the explicit in-place
    form. The missing texts (no stored key; deduplicated, first seen kept)
    are encoded by `encode_fn` with the encoder and normalization of the
    file's label, never the caller's. The fresh label must equal the
    file's in `encoder_key`, `file_sha256`, `dim` and `normalization`, or
    `StyleLabelError` names the field and nothing is written. The stored
    rows keep their fp16 values bit for bit; new rows are cast to fp16 and
    carry the `text[:80]` excerpt. The label merges: `run_name`, `date` and
    `notes`, when given, replace the old ones; the old `provenance` is
    appended to the list `provenance["previous"]` of the new one; `n_rows`,
    `norm_min` and `norm_max` are measured again over the union (a
    mixed-scale union raises). The write goes to a temp file in the folder
    of `out_path`, whose label and row count are read back before
    `os.replace`. The bar is a process crash: no fsync and no lock; one
    writer at a time is assumed.

    A side folder (`path` is a directory of parts) is extended by one new
    part, written by `save_style_embeddings` with the given free fields;
    the old parts are not rewritten. With `out_path == path` the part is
    added in place (temp file, read back, `os.replace` to the next
    `part-NNN.parquet`); with another `out_path`, which must not exist,
    the old parts are copied there first. The norm range of the union must
    stay within a factor of 2.

    Returns:
        (int, int): the stored rows before, and the rows added.
    """
    for key in free_fields:
        if key not in _FREE_FIELDS:
            raise ValueError(f"unknown free field: {key}")
    given_provenance = free_fields.get("provenance", {})
    if not isinstance(given_provenance, dict):
        raise TypeError("provenance must be a dict")
    file_label = _read_side_label(
        side_part_paths(path)[0] if os.path.isdir(path) else path)
    frame, label = load_style_embeddings(
        path, expect_encoder=file_label.get("encoder_key"))
    stored_keys = list(frame["text_key"])
    stored = set(stored_keys)
    missing_texts, missing_keys, seen = [], [], set()
    for text in texts:
        key = style_text_key(text)
        if key in stored or key in seen:
            continue
        seen.add(key)
        missing_texts.append(text)
        missing_keys.append(key)
    n_before = len(stored_keys)
    if not missing_texts:
        return n_before, 0
    if os.path.isdir(path):
        return _extend_side_folder(missing_texts, path, label, n_before,
                                   out_path=out_path, encode_fn=encode_fn,
                                   free_fields=free_fields)
    fresh, _ = _encode_for_label(missing_texts, label, encode_fn, str(path),
                                 "extend_style_embeddings")
    dim = int(label["dim"])
    old_rows = (np.stack(frame["style_emb"].to_numpy()).astype(np.float16,
                                                                copy=False)
                if n_before else np.zeros((0, dim), dtype=np.float16))
    union = np.concatenate([old_rows, np.asarray(fresh, dtype=np.float16)])
    del old_rows
    measured = measure_normalization(union)
    if measured["normalization"] != label["normalization"]:
        raise StyleLabelError(
            f"extend_style_embeddings: the union measures "
            f"{measured['normalization']!r}, the file says "
            f"{label['normalization']!r}")
    new_label = dict(label)
    new_label.update(measured)
    new_label["n_rows"] = int(union.shape[0])
    for key in ("run_name", "date", "notes"):
        if key in free_fields:
            new_label[key] = free_fields[key]
    old_provenance = label.get("provenance")
    provenance = dict(given_provenance)
    if old_provenance is not None:
        old_record = dict(old_provenance)
        chain = list(old_record.pop("previous", []))
        provenance["previous"] = chain + [old_record]
    new_label["provenance"] = provenance
    excerpts = list(frame["text_excerpt"]) + [text[:80] for text in missing_texts]
    keys = stored_keys + missing_keys
    del frame
    folder = os.path.dirname(os.path.abspath(out_path))
    handle, temp_path = tempfile.mkstemp(
        prefix=".style_extend_", suffix=".parquet.tmp", dir=folder)
    os.close(handle)
    try:
        _write_side_table(temp_path, keys, excerpts, union, new_label)
        written = _read_side_label(temp_path)
        if written != json.loads(json.dumps(new_label)) or \
                pq.ParquetFile(temp_path).metadata.num_rows != len(keys):
            raise StyleLabelError(
                f"extend_style_embeddings: the temp file {temp_path} does "
                "not read back as written")
        os.replace(temp_path, out_path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return n_before, len(missing_texts)


def _extend_side_folder(missing_texts, path, label, n_before, *, out_path,
                        encode_fn, free_fields):
    """`extend_style_embeddings` for a side folder: one new part."""
    fresh, encoded = _encode_for_label(missing_texts, label, encode_fn,
                                       str(path), "extend_style_embeddings")
    fresh_norms = _row_norms_float64(np.asarray(fresh, dtype=np.float16))
    norm_min = min(float(label["norm_min"]), float(fresh_norms.min()))
    norm_max = max(float(label["norm_max"]), float(fresh_norms.max()))
    if norm_min > 0 and norm_max / norm_min > 2:
        raise StyleLabelError(
            f"extend_style_embeddings: mixed-scale union with {path} "
            f"(min {norm_min}, max {norm_max})")
    same_folder = os.path.abspath(out_path) == os.path.abspath(path)
    if not same_folder:
        if os.path.exists(out_path):
            raise FileExistsError(f"out_path exists: {out_path}")
        os.makedirs(out_path)
        for part in side_part_paths(path):
            shutil.copy2(part, os.path.join(out_path, os.path.basename(part)))
    numbers = [int(os.path.basename(p)[len("part-"):-len(".parquet")])
               for p in side_part_paths(out_path)
               if os.path.basename(p)[len("part-"):-len(".parquet")].isdigit()]
    target = os.path.join(out_path, f"part-{max(numbers, default=-1) + 1:03d}.parquet")
    handle, temp_path = tempfile.mkstemp(
        prefix=".style_extend_", suffix=".parquet.tmp", dir=out_path)
    os.close(handle)
    try:
        save_style_embeddings(missing_texts, encoded, temp_path, **free_fields)
        written = _read_side_label(temp_path)
        if written["n_rows"] != len(missing_texts) or \
                pq.ParquetFile(temp_path).metadata.num_rows != len(missing_texts):
            raise StyleLabelError(
                f"extend_style_embeddings: the temp file {temp_path} does "
                "not read back as written")
        os.replace(temp_path, target)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return n_before, len(missing_texts)


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
