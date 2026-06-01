"""
Tests for calc_style_embeddings normalization contract.

Run on remote:
    ssh tallin.vpn 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tst311 && \\
        cd /home/abragin/src/textprism/ && \\
        pytest tst_utils/eval/metrics/test_style.py -v'
"""

import numpy as np
import pytest

from tst_utils.eval.metrics.style import calc_style_embeddings


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
