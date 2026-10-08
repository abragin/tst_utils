"""Tests for style_text_key and measure_normalization.

No GPU / no model downloads: every fixture here is a small synthesized array.
"""

import numpy as np
import os
import pandas as pd
import pytest

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
    frame, label = load_style_embeddings(path)
    assert label["n_rows"] == 3
    assert label["run_name"] == "probe"
    assert len(frame) == 3
    df = pd.DataFrame({"text": texts})
    joined = join_style_embeddings(df, path, "text", "emb")
    expected = encoded.embeddings.astype(np.float16).astype(np.float32)
    for got, want in zip(joined["emb"], expected):
        assert got.dtype == np.float32 and got.ndim == 1
        assert float(np.max(np.abs(got - want))) == 0.0


def test_duplicate_texts_stored_once(tmp_path):
    texts = ["a", "b", "a"]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, encoded, path)
    _frame, label = load_style_embeddings(path)
    assert label["n_rows"] == 2
    df = pd.DataFrame({"text": texts})
    joined = join_style_embeddings(df, path, "text", "emb")
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
        load_style_embeddings(path)


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
        load_style_embeddings(path)


def test_missing_text_raises(tmp_path):
    texts = ["stored text"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts), path)
    df = pd.DataFrame({"text": ["stored text", "absent text"]})
    with pytest.raises(StyleLabelError, match="1 texts.*absent text"):
        join_style_embeddings(df, path, "text", "emb")


def test_existing_out_col_raises(tmp_path):
    texts = ["a"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts), path)
    df = pd.DataFrame({"text": texts, "emb": [0]})
    with pytest.raises(ValueError, match="already exists"):
        join_style_embeddings(df, path, "text", "emb")


def test_join_does_not_mutate(tmp_path):
    texts = ["a", "b"]
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _handmade_encoded(texts), path)
    df = pd.DataFrame({"text": texts})
    before = df.copy(deep=True)
    joined = join_style_embeddings(df, path, "text", "emb")
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
                                   "text", "emb")
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
    _frame, label = load_style_embeddings(path)
    del label["normalization"]
    bad = str(tmp_path / "missing.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="missing required field"):
        load_style_embeddings(bad)


def test_side_file_wrong_n_rows_raises(monkeypatch, tmp_path):
    path = _valid_side_file(monkeypatch, tmp_path)
    _frame, label = load_style_embeddings(path)
    label["n_rows"] = label["n_rows"] + 1
    bad = str(tmp_path / "rows.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="n_rows"):
        load_style_embeddings(bad)


def test_side_file_wrong_dim_raises(monkeypatch, tmp_path):
    path = _valid_side_file(monkeypatch, tmp_path)
    _frame, label = load_style_embeddings(path)
    label["dim"] = label["dim"] + 1
    bad = str(tmp_path / "dim.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="width"):
        load_style_embeddings(bad)


def test_side_file_wrong_dim_still_reads_label_first(monkeypatch, tmp_path):
    """The identity checks run before any row is read, so a tampered label
    raises even when row reading is impossible (the existing guarantee)."""
    path = _valid_side_file(monkeypatch, tmp_path)
    _frame, label = load_style_embeddings(path)
    label["file_sha256"] = {"model.safetensors": "0" * 64}
    bad = str(tmp_path / "hash.parquet")
    _rewrite_label(path, bad, label)
    with pytest.raises(StyleLabelError, match="file_sha256"):
        load_style_embeddings(bad)


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


def test_centroid_round_trip(tmp_path):
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    from tst_utils.eval.data.style_store import build_centroids
    texts = ["t%d" % i for i in range(6)]
    encoded = _handmade_encoded(texts)
    path = str(tmp_path / "c.npz")
    centroids, _label = build_centroids(encoded, _handmade_groups(),
                                        renormalize=False, path=path)
    loaded = load_centroids_npz(path, renormalize=False)
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
        load_centroids_npz(p_other, renormalize=False)
    with pytest.raises(StyleLabelError, match="file_sha256"):
        load_centroids_npz(p_bad, renormalize=False)


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
        load_centroids_npz(path, renormalize=False)


def test_centroid_wrong_dim_raises(tmp_path):
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=61)          # vector is width 60
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    with pytest.raises(StyleLabelError, match="width 60"):
        load_centroids_npz(path, renormalize=False)


def test_centroid_input_normalization_unknown_allowed(tmp_path):
    import json
    from tst_utils.eval.data.load import load_centroids_npz
    label = _centroid_label(dim=60, input_normalization="unknown")
    path = str(tmp_path / "c.npz")
    np.savez(path, k=np.ones(60, dtype=np.float32),
             __style_label__=json.dumps(label))
    assert set(load_centroids_npz(path, renormalize=False)) == {"k"}


def _unlabelled_npz(path):
    np.savez(path, k=np.ones(60, dtype=np.float32))
    return path


def test_unlabelled_warns_by_default(tmp_path):
    from tst_utils.eval.data.load import load_centroids_npz
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    with pytest.warns(StyleProvenanceWarning):
        loaded = load_centroids_npz(path, renormalize=False)
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
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    monkeypatch.setattr(load_module, "STYLE_ENCODER_KEY", "base_v2")
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_module.load_centroids_npz(path, renormalize=False)
    monkeypatch.setattr(load_module, "STYLE_ENCODER_KEY", "base_v1")
    with pytest.warns(StyleProvenanceWarning):
        load_module.load_centroids_npz(path, renormalize=False)


def test_unlabelled_warning_asserts_second_emission(tmp_path):
    from tst_utils.eval.data.load import load_centroids_npz
    path = _unlabelled_npz(str(tmp_path / "u.npz"))
    load_centroids_npz(path, renormalize=False)
    with pytest.warns(StyleProvenanceWarning):
        load_centroids_npz(path, renormalize=False)


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
        load_author_styles,
        read_style_label,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        original = load_author_styles()
    assert len(original) == 5
    canonical = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "author_styles.npz")
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


def test_guard_joins_side_file(tmp_path):
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


def test_guard_side_file_missing_text_raises(tmp_path):
    import tst_utils.eval.data.style_store as store_module
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(["alpha"], _handmade_encoded(["alpha"]), path)
    df = pd.DataFrame({"text": ["alpha", "gamma"]})
    with pytest.raises(StyleLabelError, match="1 texts"):
        store_module.guard_inline_style_columns(
            df, source_style_path=path, entry_point="probe")


def test_guard_target_warns_on_base_v1():
    import tst_utils.eval.data.style_store as store_module
    df = pd.DataFrame({"text": ["alpha"],
                       "target_style_emb": [np.ones(4, dtype=np.float32)]})
    with pytest.warns(StyleProvenanceWarning, match="target_style_emb"):
        guarded = store_module.guard_inline_style_columns(df, entry_point="probe")
    assert "target_style_emb" in guarded.columns


def test_guard_target_raises_on_moved_pin(monkeypatch):
    import tst_utils.eval.data.style_store as store_module
    monkeypatch.setattr(store_module, "STYLE_ENCODER_KEY", "base_v2")
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
