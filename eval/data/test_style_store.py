"""Tests for style_text_key and measure_normalization.

No GPU / no model downloads: every fixture here is a small synthesized array.
"""

import numpy as np
import pytest

from tst_utils.eval.data.style_store import (
    StyleProvenanceWarning,
    measure_normalization,
    style_text_key,
)


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
