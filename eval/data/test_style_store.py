"""Tests for style_text_key and measure_normalization.

No GPU / no model downloads: every fixture here is a small synthesized array.
"""

import numpy as np
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
    embeddings = (
        np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)
        + scale
    )
    label = {
        "encoder_key": key,
        "file_sha256": dict(get_encoder(key)["file_sha256"]),
        "dim": 4,
        "n_rows": len(texts),
    }
    label.update(measure_normalization(embeddings))
    return EncodedStyle(embeddings=embeddings, label=label)


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
