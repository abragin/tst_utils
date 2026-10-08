"""Tests for the style encoder registry.

No GPU / no model downloads: the registry is a JSON file of hashes, and
every check here runs against it or against small dict fixtures.
"""

import pytest

from tst_utils.eval.model_names import STYLE_ENCODER_KEY, STYLE_MODEL
from tst_utils.eval.style_encoder_registry import (
    check_encoder_entry,
    get_encoder,
    get_excluded_filenames,
)

BASE_V1_WEIGHT_SHA256 = (
    "69e300548b7bdf9dc44cac4aa334ff59f068d59072d7236aaaa4147e445480f2"
)
BASE_V2_WEIGHT_SHA256 = (
    "06ee83d8db592de0b16457c1f2b29d613f49d94eb4742346a5c90590277c1afc"
)


def test_both_entries_load_with_weight_hashes():
    assert (
        get_encoder("base_v1")["file_sha256"]["model.safetensors"]
        == BASE_V1_WEIGHT_SHA256
    )
    assert (
        get_encoder("base_v2")["file_sha256"]["model.safetensors"]
        == BASE_V2_WEIGHT_SHA256
    )


def test_unknown_key_raises_with_known_keys():
    with pytest.raises(KeyError, match="base_v1"):
        get_encoder("no_such_encoder")


def test_exclude_list_reaches_past_step_1():
    assert get_excluded_filenames() == ["README.md", ".gitattributes"]


def _valid_entry():
    return {
        "hub_id": "abragin/ruBert-style-base",
        "revision": "cac972ca8a304086c4f15611a2ab923932b2cb85",
        "file_sha256": {"model.safetensors": BASE_V1_WEIGHT_SHA256},
        "description": "fixture entry",
    }


def test_schema_rejects_entry_without_hub_id_or_local_only():
    entry = _valid_entry()
    del entry["hub_id"]
    with pytest.raises(ValueError, match="hub_id"):
        check_encoder_entry("fixture", entry)


def test_schema_accepts_local_only_entry_without_hub_id():
    entry = _valid_entry()
    del entry["hub_id"]
    entry["local_only"] = "tallin"
    assert check_encoder_entry("fixture", entry) is entry


def test_model_pin_names_registry_key():
    assert STYLE_MODEL == get_encoder(STYLE_ENCODER_KEY)["hub_id"]
