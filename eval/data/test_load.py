"""
Tests for centroid loaders.

Run on remote:
    ssh tallin.vpn 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tst311 && \\
        cd /home/abragin/src/textprism/ && \\
        pytest tst_utils/eval/data/test_load.py -v'
"""

import numpy as np
import pytest

from tst_utils.eval.data.load import (
    renormalize_centroid, load_centroids_npz,
)


def test_renormalize_centroid_1d():
    v = np.array([3.0, 4.0])  # norm 5
    out = renormalize_centroid(v)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_renormalize_centroid_2d_batch():
    v = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    out = renormalize_centroid(v)
    norms = np.linalg.norm(out, axis=-1)
    assert np.allclose(norms, [1.0, 1.0, 1.0], atol=1e-6)


def test_renormalize_sub_unit_centroid():
    # Centroid of two unit vectors with cos = 0.5 has norm sqrt((1 + 0.5)/2) ≈ 0.866.
    v = np.array([0.5, 0.5*np.sqrt(3)/1])  # placeholder
    a = np.array([1.0, 0.0])
    b = np.array([0.5, np.sqrt(3)/2])  # cos(60deg) with a = 0.5
    centroid = (a + b) / 2
    pre_norm = float(np.linalg.norm(centroid))
    assert pre_norm < 0.95, f'expected sub-unit, got {pre_norm}'
    out = renormalize_centroid(centroid)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_load_centroids_npz_renormalize_true(tmp_path):
    p = tmp_path / 'c.npz'
    a = np.full(768, 0.5, dtype=np.float32)        # norm ≈ 13.86
    b = np.full(768, 0.02, dtype=np.float32)       # norm ≈ 0.554 (sub-unit)
    meta = np.array(['Tolstoy', 'News'])
    np.savez_compressed(p, Tolstoy_centroid=a, News_centroid=b, authors=meta)

    loaded = load_centroids_npz(p, renormalize=True)
    assert set(loaded.keys()) == {'Tolstoy_centroid', 'News_centroid', 'authors'}
    assert abs(float(np.linalg.norm(loaded['Tolstoy_centroid'])) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(loaded['News_centroid'])) - 1.0) < 1e-5
    # Metadata is left untouched.
    assert (loaded['authors'] == meta).all()


def test_load_centroids_npz_renormalize_false(tmp_path):
    p = tmp_path / 'c.npz'
    a = np.full(768, 0.5, dtype=np.float32)
    np.savez_compressed(p, c=a)
    loaded = load_centroids_npz(p, renormalize=False)
    assert abs(float(np.linalg.norm(loaded['c'])) - float(np.linalg.norm(a))) < 1e-5


def test_load_centroids_npz_renormalize_required(tmp_path):
    p = tmp_path / 'c.npz'
    np.savez_compressed(p, c=np.zeros(768, dtype=np.float32))
    with pytest.raises(TypeError, match='renormalize'):
        load_centroids_npz(p)
