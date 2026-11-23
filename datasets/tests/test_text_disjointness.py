"""Tests for the shared text-unit disjointness gate.

Run: pytest -q tst_utils/datasets/tests/test_text_disjointness.py
"""
import hashlib
import re

import pandas as pd
import pytest

from tst_utils.datasets.text_disjointness import norm_key, norm_keys_of


def _old_norm_key(text):
    """The definition `norm_key` had in `17 phase style-v2/_retrain_data/common.py`."""
    return hashlib.md5(re.sub(r'\W+', '', text.lower()).encode()).hexdigest()


NORM_KEY_CASES = [
    "Обычный русский текст",
    "ОБЫЧНЫЙ РУССКИЙ ТЕКСТ",
    "обычный   русский\tтекст",
    "Обычный, русский: текст!",
    "plain latin text",
    "MiXeD CaSe Latin",
    "ёжик в тумане",
    "ЕЖИК В ТУМАНЕ",
    "1234567890",
    "snake_case_name",
    "",
]


@pytest.mark.parametrize("text", NORM_KEY_CASES)
def test_norm_key_matches_the_old_definition(text):
    assert norm_key(text) == _old_norm_key(text)


def test_norm_key_collapses_punctuation_and_case():
    assert norm_key("Привет, мир!") == norm_key("привет   МИР")
    assert norm_key("Привет, мир!") != norm_key("привет мир два")


def _write(path, **cols):
    pd.DataFrame(cols).to_parquet(path, index=False)
    return str(path)


def test_norm_keys_of_two_sources_union_and_counts(tmp_path):
    a = _write(tmp_path / "a.parquet", text=["one", "two", None])
    b = _write(tmp_path / "b.parquet", text=["two", "three"])

    keys, counts = norm_keys_of([(a, ["text"]), (b, ["text"])])

    assert keys == {norm_key("one"), norm_key("two"), norm_key("three")}
    assert len(counts) == 2
    assert counts[0]["path"] == a
    assert counts[0]["columns"] == ["text"]
    assert counts[0]["rows_read"] == 3
    assert counts[0]["rows_kept"] == 3      # the null is read, then dropped from the keys
    assert counts[0]["keys"] == 2           # the null contributes nothing
    assert counts[0]["union_after"] == 2
    assert counts[1]["path"] == b
    assert counts[1]["rows_read"] == 2
    assert counts[1]["rows_kept"] == 2
    assert counts[1]["keys"] == 2
    assert counts[1]["union_after"] == 3    # "two" was already there


def test_norm_keys_of_two_text_columns(tmp_path):
    p = _write(tmp_path / "pairs.parquet",
               text_a=["left one", "left two"], text_b=["right one", None])

    keys, counts = norm_keys_of([(p, ["text_a", "text_b"])])

    assert keys == {norm_key("left one"), norm_key("left two"), norm_key("right one")}
    assert counts[0]["keys"] == 3
    assert counts[0]["rows_read"] == 2


def test_norm_keys_of_row_filter_keeps_a_subset(tmp_path):
    p = _write(tmp_path / "f.parquet",
               text=["kept", "dropped"], split=["train", "val"])

    keys, counts = norm_keys_of([
        (p, ["text"], (["split"], lambda df: df["split"] == "train")),
    ])

    assert keys == {norm_key("kept")}
    assert counts[0]["rows_read"] == 2
    assert counts[0]["rows_kept"] == 1
    assert counts[0]["keys"] == 1
    assert counts[0]["union_after"] == 1


def test_norm_keys_of_row_filter_reads_only_the_listed_columns(tmp_path):
    p = _write(tmp_path / "g.parquet", text=["x"], split=["train"], other=["y"])

    _, counts = norm_keys_of([
        (p, ["text"], (["split"], lambda df: df["split"] == "train")),
    ])

    assert counts[0]["columns"] == ["text"]


def test_norm_keys_of_empty_source_list():
    keys, counts = norm_keys_of([])
    assert keys == set()
    assert counts == []
