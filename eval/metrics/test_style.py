"""
Tests for calc_style_embeddings normalization contract.

Run on remote:
    ssh tallin.vpn 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tst311 && \\
        cd /home/abragin/src/textprism/ && \\
        pytest tst_utils/eval/metrics/test_style.py -v'
"""

import numpy as np
import pytest

import tst_utils.eval.metrics.style as style_module
from tst_utils.eval.metrics.style import (
    calc_style_embeddings,
    check_snapshot_hashes,
    load_style_encoder,
    StyleEncoderIntegrityError,
)
from tst_utils.eval.style_encoder_registry import get_encoder


try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False
gpu_only = pytest.mark.skipif(not HAS_CUDA,
                              reason="model-dependent test; run on tallin GPU")


_TEXTS = ['Привет мир.', 'Это второе предложение.']


@pytest.fixture(scope='module')
def normalized_embs():
    return calc_style_embeddings(_TEXTS, normalize=True)


@pytest.fixture(scope='module')
def raw_embs():
    return calc_style_embeddings(_TEXTS, normalize=False)


def test_normalize_required_kwarg():
    with pytest.raises(TypeError, match='normalize'):
        calc_style_embeddings(_TEXTS)


def test_normalize_must_be_keyword():
    # `normalize` is keyword-only — passing positionally must fail.
    with pytest.raises(TypeError):
        calc_style_embeddings(_TEXTS, True)


def test_returns_list_of_arrays(normalized_embs):
    assert isinstance(normalized_embs, list)
    assert len(normalized_embs) == len(_TEXTS)
    for e in normalized_embs:
        assert isinstance(e, np.ndarray)
        assert e.ndim == 1
        assert e.shape == (768,)


def test_normalize_true_produces_unit_norm(normalized_embs):
    norms = [float(np.linalg.norm(e)) for e in normalized_embs]
    for n in norms:
        assert abs(n - 1.0) < 1e-3, f'expected ~1.0, got {n}'


def test_normalize_false_preserves_native_scale(raw_embs):
    # `abragin/ruBert-style-base` produces vectors with norm ~15.
    norms = [float(np.linalg.norm(e)) for e in raw_embs]
    for n in norms:
        assert n > 10.0, f'expected raw norm > 10, got {n}'


def test_pandas_assignment_contract(normalized_embs):
    # The return type must remain assignable to a single pandas Series cell.
    import pandas as pd
    df = pd.DataFrame({'text': _TEXTS})
    df['emb'] = normalized_embs
    assert df['emb'].dtype == object
    assert isinstance(df['emb'].iloc[0], np.ndarray)
    assert df['emb'].iloc[0].shape == (768,)


@gpu_only
def test_changed_registry_value_raises(monkeypatch):
    # A changed registry hash makes the load raise, which proves that
    # the comparison runs against the entry on every load.
    real_entry = get_encoder("base_v1")
    bad_hashes = dict(real_entry["file_sha256"])
    bad_hashes["model.safetensors"] = "0" * 64
    bad_entry = dict(real_entry, file_sha256=bad_hashes)
    monkeypatch.setattr(style_module, "get_encoder", lambda key: bad_entry)
    with pytest.raises(StyleEncoderIntegrityError, match="model.safetensors"):
        load_style_encoder("base_v1")


def _write_snapshot_files(directory, files):
    import os
    for relative, content in files.items():
        path = os.path.join(str(directory), relative)
        os.makedirs(os.path.dirname(path) or str(directory), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)


def test_hash_check_catches_changed_byte(tmp_path):
    expected = {"a.bin": "0" * 64}
    _write_snapshot_files(tmp_path, {"a.bin": b"tampered"})
    with pytest.raises(StyleEncoderIntegrityError, match="a.bin"):
        check_snapshot_hashes(str(tmp_path), expected, [])


def test_hash_check_catches_added_file(tmp_path):
    import hashlib
    content = b"registered"
    expected = {"a.bin": hashlib.sha256(content).hexdigest()}
    _write_snapshot_files(tmp_path, {"a.bin": content, "extra.bin": b"x"})
    with pytest.raises(StyleEncoderIntegrityError, match="extra.bin"):
        check_snapshot_hashes(str(tmp_path), expected, ["README.md"])


def test_hash_check_ignores_excluded_file(tmp_path):
    import hashlib
    content = b"registered"
    expected = {"a.bin": hashlib.sha256(content).hexdigest()}
    _write_snapshot_files(tmp_path, {"a.bin": content, "README.md": b"docs"})
    computed = check_snapshot_hashes(str(tmp_path), expected, ["README.md"])
    assert computed["a.bin"] == expected["a.bin"]


def test_hash_check_catches_missing_file(tmp_path):
    import hashlib
    content = b"registered"
    expected = {"a.bin": hashlib.sha256(content).hexdigest(),
                "gone.bin": "1" * 64}
    _write_snapshot_files(tmp_path, {"a.bin": content})
    with pytest.raises(StyleEncoderIntegrityError, match="gone.bin"):
        check_snapshot_hashes(str(tmp_path), expected, [])


@gpu_only
def test_vectors_unchanged_after_load_check():
    # The 20 texts encoded before the change, through the new loader:
    # max abs difference 0.0, because the revision is the same snapshot.
    import json
    import os
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tests", "style_before_vectors.npz")
    stored = np.load(fixture, allow_pickle=True)
    texts = [str(t) for t in stored["texts"]]
    fresh = calc_style_embeddings(texts, normalize=False)
    difference = float(np.max(np.abs(
        np.asarray(fresh, dtype=np.float32)
        - stored["embeddings"].astype(np.float32)
    )))
    assert difference == 0.0, f"max abs difference {difference}"
    assert json.loads(str(stored["meta"]))["provenance"].startswith("tst_utils e056eda")


@gpu_only
def test_base_v2_snapshot_hashes_without_encode():
    from huggingface_hub import snapshot_download
    from tst_utils.eval.style_encoder_registry import get_excluded_filenames
    entry = get_encoder("base_v2")
    snapshot_path = snapshot_download(entry["hub_id"],
                                      revision=entry["revision"])
    check_snapshot_hashes(snapshot_path, entry["file_sha256"],
                          get_excluded_filenames())


def test_local_only_entry_raises_naming_machine(monkeypatch):
    fixture_entry = {
        "local_only": "tallin",
        "revision": "0" * 40,
        "file_sha256": {"model.safetensors": "0" * 64},
        "description": "fixture entry",
    }
    monkeypatch.setattr(style_module, "get_encoder",
                        lambda key: fixture_entry)
    with pytest.raises(NotImplementedError, match="tallin"):
        load_style_encoder("fixture")
