"""Tests for style_text_key and measure_normalization.

No GPU / no model downloads: every fixture here is a small synthesized array.
"""

import numpy as np
import os
import pandas as pd
import pytest

from tst_utils.eval.data import style_store
from tst_utils.eval.data.style_store import (
    StyleLabelError,
    StyleProvenanceWarning,
    join_style_embeddings,
    load_style_embeddings,
    measure_normalization,
    save_style_embeddings,
    style_text_key,
)
from tst_utils.eval.style_encoder_registry import get_encoder


def test_style_text_key_cyrillic():
    assert (
        style_text_key("Привет, мир! Это проверка ключа.")
        == "fa502256628fb4cc4d92eac55c2f244b74447700a7cccdd7195fd9e327ac776b"
    )


def _unit_rows(dtype, n=4, dim=8):
    rng = np.random.default_rng(11)
    rows = rng.normal(size=(n, dim))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_unit_rows_are_l2(dtype):
    label = measure_normalization(_unit_rows(dtype))
    assert label["normalization"] == "l2"
    assert label["norm_min"] == pytest.approx(1.0, abs=1e-3)
    assert label["norm_max"] == pytest.approx(1.0, abs=1e-3)
    assert isinstance(label["norm_min"], float)
    assert isinstance(label["norm_max"], float)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_just_over_boundary_is_none(dtype):
    label = measure_normalization(_unit_rows(dtype) * (1 + 2e-3))
    assert label["normalization"] == "none"


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_raw_scale_rows_pass(dtype):
    norms = np.array([14.5, 15.3, 16.7])
    rows = _unit_rows(dtype, n=3) * norms[:, None]
    label = measure_normalization(rows.astype(dtype))
    assert label["normalization"] == "none"
    assert label["norm_min"] == pytest.approx(14.5, rel=1e-3)
    assert label["norm_max"] == pytest.approx(16.7, rel=1e-3)


def test_straddling_array_raises():
    rows = np.vstack([_unit_rows(np.float64, n=2), 15.0 * _unit_rows(np.float64, n=2)])
    with pytest.raises(ValueError, match="mixed-scale"):
        measure_normalization(rows)


def test_ratio_above_two_raises():
    rows = np.vstack([10.0 * _unit_rows(np.float64, n=2),
                      25.0 * _unit_rows(np.float64, n=2)])
    with pytest.raises(ValueError, match="exceeds 2"):
        measure_normalization(rows)


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one row"):
        measure_normalization(np.zeros((0, 8)))


def _whole_array_norms(embeddings):
    """The norm computation of `measure_normalization` before the chunks."""
    return np.linalg.norm(np.asarray(embeddings, dtype=np.float64), axis=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
@pytest.mark.parametrize("n_rows", [1, 7, 64, 65, 200, 1000, 1003])
def test_chunked_row_norms_are_bit_identical(dtype, n_rows):
    # 64 rows per chunk: 200, 1000 and 1003 are not multiples of it.
    rng = np.random.default_rng(20261115 + n_rows)
    rows = (rng.normal(size=(n_rows, 768)) * 0.55).astype(dtype)
    chunked = style_store._row_norms_float64(rows, chunk_rows=64)
    whole = _whole_array_norms(rows)
    assert chunked.dtype == np.float64
    assert chunked.tobytes() == whole.tobytes()


def test_chunked_row_norms_non_contiguous_input_is_bit_identical():
    rng = np.random.default_rng(20261115)
    rows = rng.normal(size=(300, 96)).astype(np.float32)[::2, ::3]
    chunked = style_store._row_norms_float64(rows, chunk_rows=64)
    assert chunked.tobytes() == _whole_array_norms(rows).tobytes()


@pytest.mark.parametrize("kind", ["raw", "unit"])
def test_measure_normalization_chunked_equals_whole(monkeypatch, kind):
    rng = np.random.default_rng(20261115)
    rows = rng.normal(size=(1003, 768))
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    if kind == "raw":
        rows = rows * rng.uniform(12.0, 17.0, size=(1003, 1))
    rows = rows.astype(np.float32)
    whole = measure_normalization(rows)  # 1003 rows: one chunk by default
    monkeypatch.setattr(style_store, "_NORM_CHUNK_ROWS", 64)
    calls = []
    real_norm = np.linalg.norm
    monkeypatch.setattr(
        style_store.np.linalg, "norm",
        lambda array, **kw: calls.append(len(array)) or real_norm(array, **kw),
    )
    chunked = style_store.measure_normalization(rows)
    monkeypatch.undo()
    assert max(calls) == 64 and sum(calls) == 1003  # it did run in chunks
    assert chunked == whole
    assert chunked["normalization"] == ("none" if kind == "raw" else "l2")
    norms = _whole_array_norms(rows)
    assert chunked["norm_min"] == float(norms.min())
    assert chunked["norm_max"] == float(norms.max())


def test_measure_normalization_chunked_mixed_scale_same_error(monkeypatch):
    rng = np.random.default_rng(20261115)
    rows = rng.normal(size=(1003, 64))
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    rows[700:] *= 15.0  # unit rows and raw rows, in different chunks
    rows = rows.astype(np.float32)
    with pytest.raises(ValueError) as whole_error:
        measure_normalization(rows)
    monkeypatch.setattr(style_store, "_NORM_CHUNK_ROWS", 64)
    with pytest.raises(ValueError) as chunked_error:
        measure_normalization(rows)
    assert str(chunked_error.value) == str(whole_error.value)
    assert "mixed-scale" in str(chunked_error.value)


def test_measure_normalization_chunked_empty_raises(monkeypatch):
    monkeypatch.setattr(style_store, "_NORM_CHUNK_ROWS", 64)
    with pytest.raises(ValueError, match="at least one row"):
        measure_normalization(np.zeros((0, 8)))


def test_provenance_warning_is_future_warning():
    assert issubclass(StyleProvenanceWarning, FutureWarning)


def _handmade_encoded(texts, key="base_v1", scale=15.0):
    from tst_utils.eval.metrics.style import EncodedStyle
    n = len(texts)
    embeddings = scale + (np.arange(n * 4, dtype=np.float32).reshape(n, 4) % 7)
    label = {
        "encoder_key": key,
        "file_sha256": dict(get_encoder(key)["file_sha256"]),
        "dim": 4,
        "n_rows": n,
    }
    label.update(measure_normalization(embeddings))
    return EncodedStyle(embeddings=embeddings, label=label,
                        text_keys=tuple(style_text_key(t) for t in texts))


def test_round_trip_save_load_join(tmp_path):
    texts = ["первый текст", "второй текст", "третий текст"]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, encoded, path, run_name="probe")
    frame, label = load_style_embeddings(path, expect_encoder="base_v1")
    assert label["n_rows"] == 3
    assert label["run_name"] == "probe"
    assert len(frame) == 3
    df = pd.DataFrame({"text": texts})
    joined = join_style_embeddings(df, path, "text", "emb", expect_encoder="base_v1")
    expected = encoded.embeddings.astype(np.float16).astype(np.float32)
    for got, want in zip(joined["emb"], expected):
        assert got.dtype == np.float32 and got.ndim == 1
        assert float(np.max(np.abs(got - want))) == 0.0


def test_duplicate_texts_stored_once(tmp_path):
    texts = ["a", "b", "a"]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, encoded, path)
    _frame, label = load_style_embeddings(path, expect_encoder="base_v1")
    assert label["n_rows"] == 2
    df = pd.DataFrame({"text": texts})
    joined = join_style_embeddings(df, path, "text", "emb", expect_encoder="base_v1")
    assert len(joined) == 3
    assert float(np.max(np.abs(joined["emb"].iloc[0]
                               - joined["emb"].iloc[2]))) == 0.0


def test_save_rejects_non_encoded():
    with pytest.raises(TypeError, match="EncodedStyle"):
        save_style_embeddings(["a"], {"label": {}}, "x.parquet")


def test_save_rejects_length_mismatch(tmp_path):
    encoded = _handmade_encoded(["a", "b"])
    with pytest.raises(ValueError, match="1 texts for 2"):
        save_style_embeddings(["a"], encoded, str(tmp_path / "x.parquet"))


def test_save_rejects_unknown_free_field(tmp_path):
    texts = ["a"]
    with pytest.raises(ValueError, match="unknown free field"):
        save_style_embeddings(texts, _handmade_encoded(texts),
                              str(tmp_path / "x.parquet"), bogus=1)


def test_missing_label_raises(tmp_path):
    path = str(tmp_path / "plain.parquet")
    pd.DataFrame({"text_key": ["k"], "style_emb": [[1.0]]}).to_parquet(path)
    with pytest.raises(StyleLabelError, match="no style label"):
        load_style_embeddings(path, expect_encoder="base_v1")


def test_other_encoder_raises(tmp_path):
    texts = ["a"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts, key="base_v2"), path)
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_style_embeddings(path, expect_encoder="base_v1")


def test_changed_hash_raises(tmp_path):
    texts = ["a"]
    encoded = _handmade_encoded(texts)
    encoded.label["file_sha256"] = dict(encoded.label["file_sha256"])
    encoded.label["file_sha256"]["model.safetensors"] = "0" * 64
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, encoded, path)
    with pytest.raises(StyleLabelError, match="file_sha256"):
        load_style_embeddings(path, expect_encoder="base_v1")


def test_missing_text_raises(tmp_path):
    texts = ["stored text"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts), path)
    df = pd.DataFrame({"text": ["stored text", "absent text"]})
    with pytest.raises(StyleLabelError, match="1 texts.*absent text"):
        join_style_embeddings(df, path, "text", "emb", expect_encoder="base_v1")


def test_existing_out_col_raises(tmp_path):
    texts = ["a"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts), path)
    df = pd.DataFrame({"text": texts, "emb": [0]})
    with pytest.raises(ValueError, match="already exists"):
        join_style_embeddings(df, path, "text", "emb", expect_encoder="base_v1")


def test_join_does_not_mutate(tmp_path):
    texts = ["a", "b"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts), path)
    df = pd.DataFrame({"text": texts})
    before = df.copy(deep=True)
    joined = join_style_embeddings(df, path, "text", "emb", expect_encoder="base_v1")
    pd.testing.assert_frame_equal(df, before)
    assert df.attrs == {}
    assert "emb" in joined.columns and "emb" not in df.columns


def test_label_is_read_before_rows(tmp_path, monkeypatch):
    import pyarrow.parquet as pq_module
    texts = ["a"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts, key="base_v2"), path)

    def _boom(*args, **kwargs):
        raise RuntimeError("row data must not be touched")

    monkeypatch.setattr(pq_module, "read_table", _boom)
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_style_embeddings(path, expect_encoder="base_v1")


# ---------------- review F3: text key bound to the encoded text -------------

class _FakeStyleModel:
    """Deterministic stand-in: same row for every text, raw-scale (norm ~ 25)."""

    def encode(self, texts, show_progress_bar=True, normalize_embeddings=False):
        rows = np.tile(np.array([15.0, 14.0, 13.0, 12.0], dtype=np.float32),
                       (len(list(texts)), 1))
        if normalize_embeddings:
            rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        return rows


def _patch_fake_encoder(monkeypatch):
    """Route the real encode_style through a fake encoder (no model, no GPU)."""
    import tst_utils.eval.metrics.style as style_module
    monkeypatch.setattr(style_module, "load_style_encoder",
                        lambda key: _FakeStyleModel())


def test_encode_style_retains_input_text_keys(monkeypatch):
    _patch_fake_encoder(monkeypatch)
    from tst_utils.eval.metrics.style import encode_style
    encoded = encode_style(["alpha", "beta"], "base_v1", normalize=False)
    assert list(encoded.text_keys) == [style_text_key("alpha"),
                                       style_text_key("beta")]


def test_save_rejects_reordered_texts(monkeypatch, tmp_path):
    """The review's F3 case: saving ['beta','alpha'] for an encode of
    ['alpha','beta'] must raise, naming position 0 (before the fix it wrote
    silently and joined the vectors to the wrong texts)."""
    _patch_fake_encoder(monkeypatch)
    from tst_utils.eval.metrics.style import encode_style
    encoded = encode_style(["alpha", "beta"], "base_v1", normalize=False)
    path = str(tmp_path / "side.parquet")
    with pytest.raises(StyleLabelError, match="position 0"):
        save_style_embeddings(["beta", "alpha"], encoded, path)


def test_save_rejects_text_edited_after_encode(monkeypatch, tmp_path):
    _patch_fake_encoder(monkeypatch)
    from tst_utils.eval.metrics.style import encode_style
    encoded = encode_style(["  alpha  ", "beta"], "base_v1", normalize=False)
    path = str(tmp_path / "side.parquet")
    with pytest.raises(StyleLabelError, match="position 0"):
        save_style_embeddings(["alpha", "beta"], encoded, path)


def test_save_matching_texts_writes_and_joins(monkeypatch, tmp_path):
    _patch_fake_encoder(monkeypatch)
    from tst_utils.eval.metrics.style import encode_style
    texts = ["alpha", "beta"]
    encoded = encode_style(texts, "base_v1", normalize=False)
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, encoded, path)
    joined = join_style_embeddings(pd.DataFrame({"text": texts}), path,
                                   "text", "emb", expect_encoder="base_v1")
    expected = encoded.embeddings.astype(np.float16).astype(np.float32)
    for got, want in zip(joined["emb"], expected):
        assert float(np.max(np.abs(got - want))) == 0.0


# ---------------- review F4: a guarded reader needs a complete label --------

def _valid_side_file(monkeypatch, tmp_path):
    _patch_fake_encoder(monkeypatch)
    from tst_utils.eval.metrics.style import encode_style
    texts = ["alpha", "beta"]
    encoded = encode_style(texts, "base_v1", normalize=False)
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, encoded, path)
    return path


def _rewrite_label(src, dst, label):
    import json
    import pyarrow.parquet as pq
    table = pq.read_table(src)
    metadata = dict(table.schema.metadata)
    metadata[b"style_label"] = json.dumps(label).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), dst)


def test_side_file_missing_required_field_raises(monkeypatch, tmp_path):
    path = _valid_side_file(monkeypatch, tmp_path)
    _frame, label = load_style_embeddings(path, expect_encoder="base_v1")
    del label["normalization"]
    bad = str(tmp_path / "missing.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="missing required field"):
        load_style_embeddings(bad, expect_encoder="base_v1")


def test_side_file_wrong_n_rows_raises(monkeypatch, tmp_path):
    path = _valid_side_file(monkeypatch, tmp_path)
    _frame, label = load_style_embeddings(path, expect_encoder="base_v1")
    label["n_rows"] = label["n_rows"] + 1
    bad = str(tmp_path / "rows.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="n_rows"):
        load_style_embeddings(bad, expect_encoder="base_v1")


def test_side_file_wrong_dim_raises(monkeypatch, tmp_path):
    path = _valid_side_file(monkeypatch, tmp_path)
    _frame, label = load_style_embeddings(path, expect_encoder="base_v1")
    label["dim"] = label["dim"] + 1
    bad = str(tmp_path / "dim.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="width"):
        load_style_embeddings(bad, expect_encoder="base_v1")


def test_side_file_wrong_dim_still_reads_label_first(monkeypatch, tmp_path):
    """The identity checks run before any row is read, so a tampered label
    raises even when row reading is impossible (the existing guarantee)."""
    path = _valid_side_file(monkeypatch, tmp_path)
    _frame, label = load_style_embeddings(path, expect_encoder="base_v1")
    label["file_sha256"] = {"model.safetensors": "0" * 64}
    bad = str(tmp_path / "hash.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="file_sha256"):
        load_style_embeddings(bad, expect_encoder="base_v1")


def _handmade_groups(n=6):
    return ["anna", "anna", "bible", "bible", "bible", "news"][:n]


def test_build_centroids_no_renormalize(tmp_path):
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["t%d" % i for i in range(6)]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "c.npz")
    centroids, label = build_centroids(encoded, _handmade_groups(),
                                      renormalize=False, path=path,
                                      run_name="probe")
    assert set(centroids) == {"anna", "bible", "news"}
    assert all(v.dtype == np.float16 for v in centroids.values())
    assert label["kind"] == "centroid"
    assert label["encoder_key"] == "base_v1"
    assert label["renormalized"] is False
    assert label["n_rows_per_group"] == {"anna": 2, "bible": 3, "news": 1}
    assert label["input_normalization"]["normalization"] == "none"
    assert label["run_name"] == "probe"
    assert 25.0 < label["centroid_norm_min"] <= label["centroid_norm_max"] < 40.0
    for group, indices in (("anna", [0, 1]), ("bible", [2, 3, 4]),
                           ("news", [5])):
        want = encoded.embeddings[indices].astype(np.float64).mean(axis=0)
        assert float(np.max(np.abs(
            centroids[group].astype(np.float64) - want))) < 0.02


def test_build_centroids_renormalize(tmp_path):
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["t%d" % i for i in range(6)]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "c.npz")
    centroids, label = build_centroids(encoded, _handmade_groups(),
                                      renormalize=True, path=path)
    assert label["renormalized"] is True
    for vector in centroids.values():
        assert abs(float(np.linalg.norm(
            vector.astype(np.float64))) - 1.0) < 1e-3


def test_build_centroids_rejects_mixed_scale(tmp_path):
    from tst_utils.eval.data.style_store import build_centroids
    from tst_utils.eval.metrics.style import EncodedStyle
    rows = np.vstack([np.ones((2, 4), dtype=np.float32),
                      15.0 * np.ones((2, 4), dtype=np.float32)])
    label = {"encoder_key": "base_v1",
             "file_sha256": dict(get_encoder("base_v1")["file_sha256"]),
             "dim": 4, "n_rows": 4, "normalization": "none",
             "norm_min": 2.0, "norm_max": 30.0}
    encoded = EncodedStyle(embeddings=rows, label=label)
    with pytest.raises(ValueError, match="mixed-scale"):
        build_centroids(encoded, ["a", "a", "b", "b"], renormalize=False,
                        path=str(tmp_path / "c.npz"))


def test_build_centroids_rejects_label_name(tmp_path):
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["a", "b"]
    with pytest.raises(ValueError, match="__style_label__"):
        build_centroids(_handmade_encoded(texts), ["__style_label__", "b"],
                        renormalize=False, path=str(tmp_path / "c.npz"))


# ---------------- review R2-5: build_centroids binds texts by position -------

def test_build_centroids_texts_mismatch_raises(tmp_path):
    """R2-5: the same check the side-file writer uses, so a reordered text
    list cannot write the wrong centroid under a correct label."""
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["t%d" % i for i in range(6)]
    encoded = _handmade_encoded(texts)
    with pytest.raises(StyleLabelError, match="position 0"):
        build_centroids(encoded, _handmade_groups(),
                        texts=list(reversed(texts)),
                        renormalize=False, path=str(tmp_path / "c.npz"))


def test_build_centroids_matching_texts_writes(tmp_path):
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["t%d" % i for i in range(6)]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "c.npz")
    centroids, label = build_centroids(encoded, _handmade_groups(),
                                       texts=texts,
                                       renormalize=False, path=path)
    assert set(centroids) == {"anna", "bible", "news"}
    assert label["n_rows_per_group"] == {"anna": 2, "bible": 3, "news": 1}


# ---------------- task 5.2: a row may belong to several groups ---------------

def test_build_centroids_pooled_group_matches_its_member_rows(tmp_path):
    """A pooled group (the step-4 web_informal) is the float64 mean over the
    rows that name it, and each member group keeps its own mean."""
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["t%d" % i for i in range(5)]
    encoded = _handmade_encoded(texts, scale=1.0)
    groups = ["news", ("pikabu", "web"), ("pikabu", "web"), ("ficbook", "web"),
              "news"]
    centroids, label = build_centroids(encoded, groups, renormalize=True,
                                       texts=texts,
                                       path=str(tmp_path / "c.npz"))
    assert label["n_rows_per_group"] == {"news": 2, "pikabu": 2, "web": 3,
                                         "ficbook": 1}
    rows = np.asarray(encoded.embeddings, dtype=np.float64)
    expected = rows[[1, 2, 3]].mean(axis=0)
    expected = (expected / np.linalg.norm(expected)).astype(np.float16)
    assert np.array_equal(centroids["web"], expected)
    expected_pikabu = rows[[1, 2]].mean(axis=0)
    expected_pikabu = (expected_pikabu
                       / np.linalg.norm(expected_pikabu)).astype(np.float16)
    assert np.array_equal(centroids["pikabu"], expected_pikabu)
    with np.load(str(tmp_path / "c.npz")) as stored:
        assert set(stored.files) == {"news", "pikabu", "web", "ficbook",
                                     "__style_label__"}


@pytest.mark.parametrize("bad, match", [
    ((), "at least one group"),
    (("a", "a"), "same group twice"),
    (("a", "__style_label__"), "__style_label__"),
])
def test_build_centroids_rejects_bad_membership(tmp_path, bad, match):
    from tst_utils.eval.data.style_store import build_centroids
    with pytest.raises(ValueError, match=match):
        build_centroids(_handmade_encoded(["x", "y"]), ["a", bad],
                        renormalize=False, path=str(tmp_path / "c.npz"))


def test_centroid_round_trip(tmp_path):
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["t%d" % i for i in range(6)]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "c.npz")
    centroids, _label = build_centroids(encoded, _handmade_groups(),
                                        renormalize=False, path=path)
    loaded = load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")
    assert "__style_label__" not in loaded
    assert set(loaded) == set(centroids)
    for group in centroids:
        assert np.array_equal(loaded[group], centroids[group])
    with np.load(path) as raw:
        assert json.loads(str(raw["__style_label__"]))["kind"] == "centroid"


def _centroid_label(dim=60, **overrides):
    """A complete centroid label, as build_centroids writes one."""
    label = {
        "kind": "centroid",
        "encoder_key": "base_v1",
        "file_sha256": dict(get_encoder("base_v1")["file_sha256"]),
        "dim": dim,
        "input_normalization": {"normalization": "none",
                                "norm_min": 14.0, "norm_max": 16.0},
        "renormalized": False,
        "n_rows_per_group": {"k": 4},
        "centroid_norm_min": 14.0,
        "centroid_norm_max": 16.0,
    }
    label.update(overrides)
    return label


def test_labelled_centroid_mismatch_raises(tmp_path):
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    vectors = {"k": np.ones(60, dtype=np.float32)}
    good = _centroid_label(dim=60)
    other = dict(good, encoder_key="base_v2")
    bad_hash = dict(good, file_sha256={"model.safetensors": "0" * 64})
    p_other = str(tmp_path / "other.npz")
    p_bad = str(tmp_path / "bad.npz")
    np.savez(p_other, **vectors, __style_label__=json.dumps(other))
    np.savez(p_bad, **vectors, __style_label__=json.dumps(bad_hash))
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_centroids_npz(p_other, renormalize=False, expect_encoder="base_v1")
    with pytest.raises(StyleLabelError, match="file_sha256"):
        load_centroids_npz(p_bad, renormalize=False, expect_encoder="base_v1")


def test_centroid_missing_required_field_raises(tmp_path):
    """F4: a label without input_normalization must not pass as checked."""
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=60)
    del label["input_normalization"]
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    with pytest.raises(StyleLabelError, match="missing required field"):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")


def test_centroid_wrong_dim_raises(tmp_path):
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=61)          # vector is width 60
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    with pytest.raises(StyleLabelError, match="width 60"):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")


def test_centroid_input_normalization_unknown_allowed(tmp_path):
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=60, input_normalization="unknown")
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    assert set(load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")) == {"k"}


# ---------------- review R2-4: kind and the vector/group key set -------------

def test_centroid_wrong_kind_raises(tmp_path):
    """R2-4a: `kind` is required but its value is now checked."""
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=60, kind="per_row")
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    with pytest.raises(StyleLabelError, match="kind is 'per_row'"):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")


def test_centroid_extra_group_raises(tmp_path):
    """R2-4b: a file vector absent from n_rows_per_group loads silently
    before the fix; now it is named as an extra key."""
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=60)          # n_rows_per_group == {"k": 4}
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             extra=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    with pytest.raises(StyleLabelError, match=r"extra \['extra'\]"):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")


def test_centroid_missing_group_raises(tmp_path):
    """R2-4b: a label group with no file vector is named as missing."""
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=60)
    label["n_rows_per_group"] = {"k": 4, "gone": 2}
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    with pytest.raises(StyleLabelError, match=r"missing \['gone'\]"):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")


def test_centroid_small_float_array_is_metadata(tmp_path):
    """R2-4c: a short float array beside the vectors is metadata in the label
    check AND in the renormalizer. Before the fix the dim check raised
    'has width 2, label says dim 60' on this file."""
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=60)          # n_rows_per_group == {"k": 4}
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             meta=np.array([1.0, 2.0], dtype=np.float32),
             __style_label__=json.dumps(label))
    loaded = load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")
    assert set(loaded) == {"k", "meta"}
    assert np.array_equal(loaded["meta"], np.array([1.0, 2.0], dtype=np.float32))
    with_renorm = load_centroids_npz(path, renormalize=True, expect_encoder="base_v1")
    assert np.array_equal(with_renorm["meta"],
                          np.array([1.0, 2.0], dtype=np.float32))


def _unlabelled_npz(path):
    np.savez(path, k=np.ones(60, dtype=np.float32))
    return path


def test_unlabelled_warns_by_default(tmp_path, monkeypatch):
    """The default under the base_v1 pin; the base_v2 half of the pair is
    test_unlabelled_default_follows_patched_pin."""
    from tst_utils.eval import model_names
    from tst_utils.eval.data.load import load_centroids_npz
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v1")
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    with pytest.warns(StyleProvenanceWarning):
        loaded = load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")
    assert set(loaded) == {"k"}


def test_unlabelled_warns_explicit_base_v1(tmp_path):
    from tst_utils.eval.data.load import load_centroids_npz
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    with pytest.warns(StyleProvenanceWarning):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")


def test_unlabelled_raises_for_base_v2(tmp_path):
    from tst_utils.eval.data.load import load_centroids_npz
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v2")


def test_unlabelled_default_follows_patched_pin(tmp_path, monkeypatch):
    import tst_utils.eval.data.load as load_module
    from tst_utils.eval import model_names
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v2")
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_module.load_centroids_npz(path, renormalize=False)
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v1")
    with pytest.warns(StyleProvenanceWarning):
        load_module.load_centroids_npz(path, renormalize=False)


def test_unlabelled_warning_asserts_second_emission(tmp_path):
    from tst_utils.eval.data.load import load_centroids_npz
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")
    with pytest.warns(StyleProvenanceWarning):
        load_centroids_npz(path, renormalize=False, expect_encoder="base_v1")


def test_read_style_label(tmp_path):
    import json
    from tst_utils.eval.data.load import read_style_label
    labelled = str(tmp_path / "l.npz")
    label = {"encoder_key": "base_v1"}
    np.savez(labelled, k=np.ones(4), __style_label__=json.dumps(label))
    assert read_style_label(labelled) == label
    assert read_style_label(_unlabelled_npz(str(tmp_path / "u.npz"))) is None


def test_load_author_styles_labelled_copy_matches(tmp_path):
    import json
    import warnings

    from tst_utils.eval.data.load import (
        AUTHOR_STYLES_BASE_V1_PATH,
        load_centroids_npz,
        read_style_label,
    )
    # The v1 file is the copy at its own path (task 5.2); the canonical
    # path follows the pin.
    canonical = AUTHOR_STYLES_BASE_V1_PATH
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        original = load_centroids_npz(canonical, renormalize=False,
                                      expect_encoder="base_v1")
    assert len(original) == 5
    label = read_style_label(canonical)
    assert label["encoder_key"] == "base_v1"
    assert label["file_sha256"] == get_encoder("base_v1")["file_sha256"]
    with np.load(canonical) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    assert set(arrays) == set(original) | {"__style_label__"}
    for key in original:
        assert np.array_equal(arrays[key], original[key])
    copy_path = str(tmp_path / "author_styles_copy.npz")
    np.savez(copy_path, **arrays)
    assert read_style_label(copy_path) == label


def _inline_frame():
    return pd.DataFrame({
        "text": ["alpha", "beta"],
        "text_style_emb": [np.ones(4, dtype=np.float32),
                           2 * np.ones(4, dtype=np.float32)],
    })


def test_guard_drops_inline_source_column():
    import tst_utils.eval.data.style_store as store_module
    df = _inline_frame()
    with pytest.warns(StyleProvenanceWarning, match="text_style_emb"):
        guarded = store_module.guard_inline_style_columns(df, entry_point="probe")
    assert "text_style_emb" not in guarded.columns
    assert list(guarded.columns) == ["text"]
    pd.testing.assert_frame_equal(df, _inline_frame())


def test_guard_joins_side_file(tmp_path, monkeypatch):
    from tst_utils.eval import model_names
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v1")
    import warnings
    import tst_utils.eval.data.style_store as store_module
    texts = ["alpha", "beta"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts), path)
    df = _inline_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", StyleProvenanceWarning)
        guarded = store_module.guard_inline_style_columns(
            df, source_style_path=path, entry_point="probe")
    assert guarded["text_style_emb"].iloc[0].dtype == np.float32
    assert float(np.max(np.abs(
        np.stack(guarded["text_style_emb"].to_numpy())
        - _handmade_encoded(texts).embeddings.astype(np.float16).astype(np.float32)
    ))) == 0.0


def test_guard_side_file_missing_text_raises(tmp_path, monkeypatch):
    from tst_utils.eval import model_names
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v1")
    import tst_utils.eval.data.style_store as store_module
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(["alpha"], _handmade_encoded(["alpha"]), path)
    df = pd.DataFrame({"text": ["alpha", "gamma"]})
    with pytest.raises(StyleLabelError, match="1 texts"):
        store_module.guard_inline_style_columns(
            df, source_style_path=path, entry_point="probe")


def test_guard_target_warns_on_base_v1(monkeypatch):
    from tst_utils.eval import model_names
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v1")
    import tst_utils.eval.data.style_store as store_module
    df = pd.DataFrame({"text": ["alpha"],
                       "target_style_emb": [np.ones(4, dtype=np.float32)]})
    with pytest.warns(StyleProvenanceWarning, match="target_style_emb"):
        guarded = store_module.guard_inline_style_columns(df, entry_point="probe")
    assert "target_style_emb" in guarded.columns


def test_guard_target_raises_on_moved_pin(monkeypatch):
    import tst_utils.eval.data.style_store as store_module
    from tst_utils.eval import model_names
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v2")
    df = pd.DataFrame({"text": ["alpha"],
                       "target_style_emb": [np.ones(4, dtype=np.float32)]})
    with pytest.raises(StyleLabelError, match="target_style_emb"):
        store_module.guard_inline_style_columns(df, entry_point="probe")


def test_guard_quiet_frame_stays_quiet():
    import warnings
    import tst_utils.eval.data.style_store as store_module
    df = pd.DataFrame({"text": ["alpha"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", StyleProvenanceWarning)
        guarded = store_module.guard_inline_style_columns(df, entry_point="probe")
    pd.testing.assert_frame_equal(guarded, df)


# ---------------- extend_style_embeddings and the encode mode of the join ---
# (task 2026-11-15-style-v2-full-set-recompute, step 5)

import hashlib  # noqa: E402
import logging  # noqa: E402

from tst_utils.eval.data.style_store import extend_style_embeddings  # noqa: E402


class _StubEncoder:
    """An `encode_fn` stand-in: text-dependent raw-scale rows (norm ~15).

    `label_changes` are applied to the label it returns; `dim`, `unit`,
    `scale`, `shuffle_keys` and `drop_row` fake the other failure modes.
    Every call is recorded with its arguments.
    """

    def __init__(self, dim=8, label_changes=None, unit=False, scale=15.0,
                 shuffle_keys=False, drop_row=False):
        self.dim, self.label_changes = dim, label_changes or {}
        self.unit, self.scale = unit, scale
        self.shuffle_keys, self.drop_row = shuffle_keys, drop_row
        self.calls = []

    @staticmethod
    def row(text, dim=8, scale=15.0):
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8],
                              "little")
        vector = np.random.default_rng(seed).normal(size=dim)
        return (scale * vector / np.linalg.norm(vector)
                * (1.0 + (seed % 100) / 1000.0)).astype(np.float32)

    def __call__(self, texts, encoder_key, *, normalize):
        from tst_utils.eval.metrics.style import EncodedStyle
        texts = list(texts)
        self.calls.append({"texts": texts, "encoder_key": encoder_key,
                           "normalize": normalize})
        rows = np.stack([self.row(t, self.dim, self.scale) for t in texts])
        if self.unit or normalize:
            rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        keys = [style_text_key(t) for t in texts]
        if self.shuffle_keys:
            keys = keys[::-1]
        if self.drop_row:
            rows = rows[:-1]
        label = {"encoder_key": encoder_key,
                 "file_sha256": dict(get_encoder(encoder_key)["file_sha256"]),
                 "dim": int(rows.shape[1]), "n_rows": int(rows.shape[0])}
        label.update(measure_normalization(rows))
        label.update(self.label_changes)
        return EncodedStyle(embeddings=rows, label=label,
                            text_keys=tuple(keys))


def _stub_side_file(tmp_path, texts, name="side.parquet", **free_fields):
    path = str(tmp_path / name)
    encoded = _StubEncoder()(texts, "base_v1", normalize=False)
    save_style_embeddings(texts, encoded, path, **free_fields)
    return path


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


_BASE_TEXTS = ["альфа", "бета", "гамма", "дельта"]


@pytest.mark.parametrize("field, stub", [
    ("encoder_key", _StubEncoder(label_changes={"encoder_key": "base_v2"})),
    ("file_sha256", _StubEncoder(label_changes={"file_sha256": {"x": "y"}})),
    ("dim", _StubEncoder(dim=6)),
    ("normalization", _StubEncoder(unit=True)),
])
def test_extend_refuses_a_fresh_label_mismatch(tmp_path, field, stub):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    before = _bytes(path)
    with pytest.raises(StyleLabelError, match=field):
        extend_style_embeddings(["новый"], path, out_path=path, encode_fn=stub)
    assert len(stub.calls) == 1  # encoded, then refused
    assert _bytes(path) == before
    assert sorted(os.listdir(tmp_path)) == ["side.parquet"]


@pytest.mark.parametrize("stub, match", [
    (_StubEncoder(shuffle_keys=True), "position 0"),
    (_StubEncoder(drop_row=True), "rows for 2 texts"),
])
def test_extend_refuses_other_keys_or_row_count(tmp_path, stub, match):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    before = _bytes(path)
    with pytest.raises(StyleLabelError, match=match):
        extend_style_embeddings(["новый", "второй"], path, out_path=path,
                                encode_fn=stub)
    assert _bytes(path) == before


def test_extend_derives_encoder_and_normalize_from_the_label(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    stub = _StubEncoder()
    extend_style_embeddings(["новый"], path, out_path=path, encode_fn=stub)
    assert stub.calls[0]["encoder_key"] == "base_v1"
    assert stub.calls[0]["normalize"] is False
    unit_path = str(tmp_path / "unit.parquet")
    texts = ["один", "два"]
    save_style_embeddings(texts, _StubEncoder()(texts, "base_v1", normalize=True),
                          unit_path)
    unit_stub = _StubEncoder()
    extend_style_embeddings(["три"], unit_path, out_path=unit_path,
                            encode_fn=unit_stub)
    assert unit_stub.calls[0]["normalize"] is True
    assert load_style_embeddings(unit_path, "base_v1")[1]["normalization"] == "l2"


def test_extend_appends_dedups_and_keeps_old_rows_bit_identical(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS, run_name="first",
                           date="2026-11-15", provenance={"step": 1})
    old_frame, _ = load_style_embeddings(path, "base_v1")
    stub = _StubEncoder()
    result = extend_style_embeddings(
        ["бета", "новый", "новый", "ещё", "альфа"], path,
        out_path=path, encode_fn=stub)
    assert result == (4, 2)
    assert stub.calls[0]["texts"] == ["новый", "ещё"]
    frame, label = load_style_embeddings(path, "base_v1")
    assert list(frame["text_key"]) == (list(old_frame["text_key"])
                                      + [style_text_key("новый"),
                                         style_text_key("ещё")])
    assert list(frame["text_excerpt"])[-2:] == ["новый", "ещё"]
    old = np.stack(old_frame["style_emb"].to_numpy())
    new = np.stack(frame["style_emb"].to_numpy())
    assert new.dtype == np.float16
    assert new[:4].tobytes() == old.tobytes()
    assert new[4].tobytes() == _StubEncoder.row("новый").astype(np.float16).tobytes()
    assert label["n_rows"] == 6
    norms = np.linalg.norm(new.astype(np.float64), axis=1)
    assert label["norm_min"] == float(norms.min())
    assert label["norm_max"] == float(norms.max())


def test_extend_with_nothing_missing_writes_nothing(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    before = _bytes(path)
    stub = _StubEncoder()
    out = str(tmp_path / "out.parquet")
    assert extend_style_embeddings(_BASE_TEXTS, path, out_path=out,
                                   encode_fn=stub) == (4, 0)
    assert stub.calls == [] and _bytes(path) == before
    assert not os.path.exists(out)


def test_extend_to_another_path_leaves_the_source(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    before = _bytes(path)
    out = str(tmp_path / "grown.parquet")
    assert extend_style_embeddings(["новый"], path, out_path=out,
                                   encode_fn=_StubEncoder()) == (4, 1)
    assert _bytes(path) == before
    assert load_style_embeddings(out, "base_v1")[1]["n_rows"] == 5


def test_extend_requires_out_path_and_encode_fn(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    with pytest.raises(TypeError):
        extend_style_embeddings(["новый"], path, encode_fn=_StubEncoder())
    with pytest.raises(TypeError):
        extend_style_embeddings(["новый"], path, out_path=path)
    with pytest.raises(ValueError, match="unknown free field"):
        extend_style_embeddings(["новый"], path, out_path=path,
                                encode_fn=_StubEncoder(), source="x")


def test_extend_merges_the_label_over_two_extends(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS, run_name="first",
                           date="2026-11-15", notes="n0",
                           provenance={"step": 0})
    extend_style_embeddings(["один"], path, out_path=path,
                            encode_fn=_StubEncoder(), run_name="second",
                            provenance={"step": 1})
    extend_style_embeddings(["два"], path, out_path=path,
                            encode_fn=_StubEncoder(), date="2026-11-16",
                            provenance={"step": 2})
    _, label = load_style_embeddings(path, "base_v1")
    assert label["run_name"] == "second"
    assert label["date"] == "2026-11-16"
    assert label["notes"] == "n0"
    assert label["provenance"] == {
        "step": 2, "previous": [{"step": 0}, {"step": 1}]}
    assert label["n_rows"] == 6


def test_extend_refuses_a_mixed_scale_union(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    before = _bytes(path)
    with pytest.raises(ValueError, match="mixed-scale"):
        extend_style_embeddings(["новый"], path, out_path=path,
                                encode_fn=_StubEncoder(scale=45.0))
    assert _bytes(path) == before


def test_extend_failed_write_leaves_the_original(tmp_path, monkeypatch):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    before = _bytes(path)

    def partial_write(target, *args, **kwargs):
        with open(target, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(style_store, "_write_side_table", partial_write)
    with pytest.raises(OSError, match="disk full"):
        extend_style_embeddings(["новый"], path, out_path=path,
                                encode_fn=_StubEncoder())
    assert _bytes(path) == before
    assert sorted(os.listdir(tmp_path)) == ["side.parquet"]


def test_strict_join_default_still_raises_with_the_new_signature(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    df = pd.DataFrame({"text": ["альфа", "нет такого", "нет такого"]})
    with pytest.raises(StyleLabelError, match="2 texts have no stored row"):
        join_style_embeddings(df, path, "text", "emb", expect_encoder="base_v1")


def test_joined_column_holds_no_view_of_the_whole_file(tmp_path):
    texts = [f"текст {i}" for i in range(50)]
    path = _stub_side_file(tmp_path, texts)
    df = pd.DataFrame({"text": ["текст 3", "текст 7", "текст 3"]})
    joined = join_style_embeddings(df, path, "text", "emb",
                                   expect_encoder="base_v1")
    for vector in joined["emb"]:
        assert vector.dtype == np.float32
        assert vector.base is not None and vector.base.shape[0] == 2
    assert np.array_equal(joined["emb"][0], joined["emb"][2])


def test_encode_mode_requires_encode_fn_and_a_known_mode(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    df = pd.DataFrame({"text": ["альфа"]})
    with pytest.raises(TypeError, match="encode_fn"):
        join_style_embeddings(df, path, "text", "emb", "base_v1",
                              on_missing="encode")
    with pytest.raises(ValueError, match="on_missing"):
        join_style_embeddings(df, path, "text", "emb", "base_v1",
                              on_missing="skip")


def test_encode_mode_vector_is_bit_equal_stored_and_fresh(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    df = pd.DataFrame({"text": ["альфа", "новый", "альфа"]})
    stub = _StubEncoder()
    joined = join_style_embeddings(df, path, "text", "emb", "base_v1",
                                   on_missing="encode", encode_fn=stub,
                                   max_missing_fraction=0.5)
    assert stub.calls[0]["texts"] == ["новый"]
    assert stub.calls[0]["normalize"] is False
    extend_style_embeddings(["новый"], path, out_path=path,
                            encode_fn=_StubEncoder())
    strict = join_style_embeddings(df, path, "text", "emb", "base_v1")
    for fresh, stored in zip(joined["emb"], strict["emb"]):
        assert fresh.dtype == stored.dtype == np.float32
        assert fresh.tobytes() == stored.tobytes()


def test_encode_mode_threshold(tmp_path, caplog):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    df = pd.DataFrame({"text": ["альфа", "бета", "гамма", "новый", "новый"]})
    # 1 missing of 4 unique texts: 0.25.
    with pytest.raises(StyleLabelError, match=r"1 of 4 unique texts"):
        join_style_embeddings(df, path, "text", "emb", "base_v1",
                              on_missing="encode", encode_fn=_StubEncoder(),
                              max_missing_fraction=0.2)
    caplog.set_level(logging.WARNING,
                     logger="tst_utils.eval.data.style_store")
    stub = _StubEncoder()
    joined = join_style_embeddings(df, path, "text", "emb", "base_v1",
                                   on_missing="encode", encode_fn=stub,
                                   max_missing_fraction=0.25)
    assert len(joined) == 5 and len(stub.calls) == 1
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1
    assert path in lines[0] and "1 missing of 4 unique" in lines[0]


def test_encode_mode_refuses_a_fresh_label_mismatch(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    df = pd.DataFrame({"text": ["альфа", "новый"]})
    with pytest.raises(StyleLabelError, match="file_sha256"):
        join_style_embeddings(
            df, path, "text", "emb", "base_v1", on_missing="encode",
            encode_fn=_StubEncoder(label_changes={"file_sha256": {"x": "y"}}),
            max_missing_fraction=1.0)


@pytest.mark.parametrize("mode", ["raise", "encode"])
def test_join_of_an_empty_column(tmp_path, mode):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    df = pd.DataFrame({"text": pd.Series([], dtype=object)})
    stub = _StubEncoder()
    joined = join_style_embeddings(df, path, "text", "emb", "base_v1",
                                   on_missing=mode, encode_fn=stub)
    assert len(joined) == 0 and "emb" in joined.columns and stub.calls == []


def test_preloaded_form_joins_the_same_and_still_checks(tmp_path):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    df = pd.DataFrame({"text": ["гамма", "альфа"]})
    pair = load_style_embeddings(path, "base_v1")
    by_path = join_style_embeddings(df, path, "text", "emb", "base_v1")
    preloaded = join_style_embeddings(df, pair, "text", "emb", "base_v1")
    again = join_style_embeddings(df, pair, "text", "emb", "base_v1")
    for a, b, c in zip(by_path["emb"], preloaded["emb"], again["emb"]):
        assert a.tobytes() == b.tobytes() == c.tobytes()
    with pytest.raises(StyleLabelError, match="expected 'base_v2'"):
        join_style_embeddings(df, pair, "text", "emb", "base_v2")
    frame, label = pair
    with pytest.raises(StyleLabelError, match="n_rows"):
        join_style_embeddings(df, (frame.iloc[:3], label), "text", "emb",
                              "base_v1")
    with pytest.raises(TypeError, match="pair"):
        join_style_embeddings(df, (frame,), "text", "emb", "base_v1")


def test_preloaded_index_is_built_once_per_frame(tmp_path, monkeypatch):
    path = _stub_side_file(tmp_path, _BASE_TEXTS)
    pair = load_style_embeddings(path, "base_v1")
    df = pd.DataFrame({"text": ["альфа"]})
    builds = []
    real = style_store._preloaded_index

    def counting(frame):
        before = id(frame) in style_store._PRELOADED_INDEX
        builds.append(not before)
        return real(frame)
    monkeypatch.setattr(style_store, "_preloaded_index", counting)
    for _ in range(3):
        join_style_embeddings(df, pair, "text", "emb", "base_v1")
    assert builds == [True, False, False]


# ---------------- side folders of parts (host, 2026-11-15) ------------------

from tst_utils.eval.data.style_store import side_part_paths  # noqa: E402


def _stub_folder(tmp_path, parts, name="side", stubs=None):
    folder = tmp_path / name
    folder.mkdir()
    for number, texts in enumerate(parts):
        stub = (stubs or {}).get(number, _StubEncoder())
        encoded = stub(texts, "base_v1", normalize=False)
        save_style_embeddings(texts, encoded,
                              str(folder / f"part-{number:03d}.parquet"),
                              run_name=f"part {number}")
    return str(folder)


_PARTS = [["альфа", "бета"], ["гамма", "дельта", "эпсилон"], ["дзета"]]


def test_folder_loads_all_parts_in_order(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS)
    frame, label = load_style_embeddings(folder, "base_v1")
    texts = [t for part in _PARTS for t in part]
    assert list(frame["text_key"]) == [style_text_key(t) for t in texts]
    assert label["n_rows"] == 6 and label["encoder_key"] == "base_v1"
    assert [p["file"] for p in label["parts"]] == [
        "part-000.parquet", "part-001.parquet", "part-002.parquet"]
    assert [p["run_name"] for p in label["parts"]] == ["part 0", "part 1", "part 2"]
    assert label["norm_min"] == min(p["norm_min"] for p in label["parts"])
    assert label["norm_max"] == max(p["norm_max"] for p in label["parts"])


def test_folder_join_strict_and_preloaded(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS)
    single = _stub_side_file(tmp_path, [t for part in _PARTS for t in part])
    df = pd.DataFrame({"text": ["дзета", "альфа", "гамма"]})
    from_folder = join_style_embeddings(df, folder, "text", "emb", "base_v1")
    from_file = join_style_embeddings(df, single, "text", "emb", "base_v1")
    pair = load_style_embeddings(folder, "base_v1")
    preloaded = join_style_embeddings(df, pair, "text", "emb", "base_v1")
    for a, b, c in zip(from_folder["emb"], from_file["emb"], preloaded["emb"]):
        assert a.tobytes() == b.tobytes() == c.tobytes()
    with pytest.raises(StyleLabelError, match="no stored row"):
        join_style_embeddings(pd.DataFrame({"text": ["нет"]}), folder,
                              "text", "emb", "base_v1")


@pytest.mark.parametrize("stub, match", [
    (_StubEncoder(unit=True), "normalization"),
    (_StubEncoder(dim=6), "dim"),
])
def test_folder_refuses_parts_that_differ(tmp_path, stub, match):
    folder = _stub_folder(tmp_path, _PARTS, stubs={2: stub})
    with pytest.raises(StyleLabelError, match=match):
        load_style_embeddings(folder, "base_v1")


def test_folder_refuses_a_part_of_another_encoder(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS)
    other = _StubEncoder()(["омега"], "base_v2", normalize=False)
    save_style_embeddings(["омега"], other, os.path.join(folder, "part-003.parquet"))
    with pytest.raises(StyleLabelError, match="part-003.parquet names encoder"):
        load_style_embeddings(folder, "base_v1")


def test_folder_refuses_a_key_in_two_parts(tmp_path):
    folder = _stub_folder(tmp_path, [["альфа", "бета"], ["бета"]])
    with pytest.raises(StyleLabelError, match="more than one part"):
        load_style_embeddings(folder, "base_v1")


def test_folder_refuses_mixed_scale_parts(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS, stubs={1: _StubEncoder(scale=45.0)})
    with pytest.raises(StyleLabelError, match="mixed-scale"):
        load_style_embeddings(folder, "base_v1")


def test_empty_folder_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(StyleLabelError, match="no part"):
        load_style_embeddings(str(tmp_path / "empty"), "base_v1")


def test_folder_extend_in_place_adds_one_part(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS)
    before = {p: _bytes(p) for p in side_part_paths(folder)}
    stub = _StubEncoder()
    result = extend_style_embeddings(["бета", "новый", "новый", "ещё"], folder,
                                     out_path=folder, encode_fn=stub,
                                     run_name="extend 1", provenance={"why": "t"})
    assert result == (6, 2)
    assert stub.calls[0]["texts"] == ["новый", "ещё"]
    assert stub.calls[0]["normalize"] is False
    for part, content in before.items():
        assert _bytes(part) == content
    assert sorted(os.listdir(folder)) == [
        "part-000.parquet", "part-001.parquet", "part-002.parquet",
        "part-003.parquet"]
    frame, label = load_style_embeddings(folder, "base_v1")
    assert label["n_rows"] == 8
    assert label["parts"][3]["run_name"] == "extend 1"
    assert label["parts"][3]["provenance"] == {"why": "t"}
    assert list(frame["text_excerpt"])[-2:] == ["новый", "ещё"]


def test_folder_extend_to_another_folder_leaves_the_source(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS)
    before = {p: _bytes(p) for p in side_part_paths(folder)}
    out = str(tmp_path / "grown")
    assert extend_style_embeddings(["новый"], folder, out_path=out,
                                   encode_fn=_StubEncoder()) == (6, 1)
    assert {p: _bytes(p) for p in side_part_paths(folder)} == before
    assert sorted(os.listdir(folder)) == sorted(os.path.basename(p) for p in before)
    assert load_style_embeddings(out, "base_v1")[1]["n_rows"] == 7
    with pytest.raises(FileExistsError):
        extend_style_embeddings(["другой"], folder, out_path=out,
                                encode_fn=_StubEncoder())


@pytest.mark.parametrize("stub, match", [
    (_StubEncoder(label_changes={"file_sha256": {"x": "y"}}), "file_sha256"),
    (_StubEncoder(unit=True), "normalization"),
    (_StubEncoder(scale=45.0), "mixed-scale"),
])
def test_folder_extend_refusal_writes_nothing(tmp_path, stub, match):
    folder = _stub_folder(tmp_path, _PARTS)
    before = {p: _bytes(p) for p in side_part_paths(folder)}
    with pytest.raises(StyleLabelError, match=match):
        extend_style_embeddings(["новый"], folder, out_path=folder,
                                encode_fn=stub)
    assert sorted(os.listdir(folder)) == sorted(os.path.basename(p) for p in before)
    assert {p: _bytes(p) for p in side_part_paths(folder)} == before


def test_folder_extend_failed_write_leaves_no_part(tmp_path, monkeypatch):
    folder = _stub_folder(tmp_path, _PARTS)
    names = sorted(os.listdir(folder))

    def failing_save(texts, encoded, path, **free_fields):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(style_store, "save_style_embeddings", failing_save)
    with pytest.raises(OSError, match="disk full"):
        extend_style_embeddings(["новый"], folder, out_path=folder,
                                encode_fn=_StubEncoder())
    assert sorted(os.listdir(folder)) == names


def test_folder_encode_mode_join(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS)
    df = pd.DataFrame({"text": ["альфа", "новый"]})
    stub = _StubEncoder()
    joined = join_style_embeddings(df, folder, "text", "emb", "base_v1",
                                   on_missing="encode", encode_fn=stub,
                                   max_missing_fraction=0.5)
    assert stub.calls[0]["texts"] == ["новый"]
    assert joined["emb"][1].tobytes() == _StubEncoder.row("новый").astype(
        np.float16).astype(np.float32).tobytes()


# ------- the join by path reads only the rows it needs (host, 2026-11-15) ---

def _relabel(path, **changes):
    import json
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    label = json.loads(metadata[b"style_label"])
    label.update(changes)
    metadata[b"style_label"] = json.dumps(label).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


def test_path_join_does_not_load_the_whole_file(tmp_path, monkeypatch):
    folder = _stub_folder(tmp_path, _PARTS)
    single = _stub_side_file(tmp_path, [t for part in _PARTS for t in part])
    df = pd.DataFrame({"text": ["дзета", "альфа", "дзета", "гамма"]})
    expected = [join_style_embeddings(df, load_style_embeddings(path, "base_v1"),
                                      "text", "emb", "base_v1")
                for path in (folder, single)]

    def no_load(*args, **kwargs):
        raise AssertionError("the join by path loaded the whole file")
    monkeypatch.setattr(style_store, "load_style_embeddings", no_load)
    for path, want in zip((folder, single), expected):
        joined = join_style_embeddings(df, path, "text", "emb", "base_v1")
        for got, row in zip(joined["emb"], want["emb"]):
            assert got.dtype == np.float32 and got.tobytes() == row.tobytes()


def test_path_join_reads_vectors_only_of_the_parts_it_needs(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    folder = _stub_folder(tmp_path, _PARTS)
    reads = []
    real_read_table = pq.read_table

    def counting(path, *args, columns=None, **kwargs):
        reads.append((os.path.basename(str(path)), tuple(columns or ())))
        return real_read_table(path, *args, columns=columns, **kwargs)
    monkeypatch.setattr(pq, "read_table", counting)
    df = pd.DataFrame({"text": ["гамма", "эпсилон"]})
    joined = join_style_embeddings(df, folder, "text", "emb", "base_v1")
    assert len(joined) == 2
    assert [name for name, columns in reads if columns != ("text_key",)] == [
        "part-001.parquet"]
    assert {columns for _, columns in reads} == {("text_key",), ("style_emb",)}


@pytest.mark.parametrize("parts, stubs, match", [
    ([["альфа", "бета"], ["бета"]], None, "more than one part"),
    (_PARTS, {1: _StubEncoder(scale=45.0)}, "mixed-scale"),
    (_PARTS, {2: _StubEncoder(unit=True)}, "normalization"),
    (_PARTS, {2: _StubEncoder(dim=6)}, "dim"),
])
def test_path_join_keeps_the_folder_checks(tmp_path, parts, stubs, match):
    folder = _stub_folder(tmp_path, parts, stubs=stubs)
    with pytest.raises(StyleLabelError, match=match):
        join_style_embeddings(pd.DataFrame({"text": ["альфа"]}), folder,
                              "text", "emb", "base_v1")


def test_path_join_checks_each_label_against_its_file(tmp_path):
    folder = _stub_folder(tmp_path, _PARTS)
    _relabel(side_part_paths(folder)[1], n_rows=7)
    with pytest.raises(StyleLabelError, match="label says n_rows 7 but the file has 3"):
        join_style_embeddings(pd.DataFrame({"text": ["альфа"]}), folder,
                              "text", "emb", "base_v1")
    single = _stub_side_file(tmp_path, ["альфа", "бета"])
    _relabel(single, dim=5)
    with pytest.raises(StyleLabelError, match="dim"):
        join_style_embeddings(pd.DataFrame({"text": ["альфа"]}), single,
                              "text", "emb", "base_v1")


def test_path_join_reads_every_label_before_any_row(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    folder = _stub_folder(tmp_path, _PARTS)
    other = _StubEncoder()(["омега"], "base_v2", normalize=False)
    save_style_embeddings(["омега"], other, os.path.join(folder, "part-003.parquet"))

    def boom(*args, **kwargs):
        raise RuntimeError("row data must not be touched")
    monkeypatch.setattr(pq, "read_table", boom)
    with pytest.raises(StyleLabelError, match="part-003.parquet names encoder"):
        join_style_embeddings(pd.DataFrame({"text": ["альфа"]}), folder,
                              "text", "emb", "base_v1")
