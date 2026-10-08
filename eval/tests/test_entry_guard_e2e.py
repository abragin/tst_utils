"""End-to-end guard test: TstPerformanceMetrics.execute with a canned tst_func.

1D (one output per input) and 2D (two outputs per input). The style
encoder is monkeypatched to a counter that returns fixed vectors; the
other scorers (perplexity, BERTScore, LaBSE) run for real, so this file
is gpu_only and runs on tallin.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tst_utils.eval.data.style_store import StyleProvenanceWarning
from tst_utils.eval.performance import TstPerformanceMetrics

HAS_CUDA = torch.cuda.is_available()
gpu_only = pytest.mark.skipif(
    not HAS_CUDA, reason="other scorers run for real — run on tallin GPU")

SOURCES = [
    "Сегодня утром в городе прошёл сильный дождь.",
    "Учёные представили новое исследование о климате.",
    "Мальчик медленно шёл по пустой улице.",
]
STYLED = [
    "Поутру над городом разразился проливной дождь.",
    "Мужи науки явили миру новый труд о климате.",
    "Отрок неспешно брёл по опустевшей улице.",
]
STYLED_V1 = [
    "Утром на город обрушился сильный ливень.",
    "Исследователи обнародовали свежую работу о климате.",
    "Юноша тихо шагал вдоль безлюдной улицы.",
]
TARGETS = ["Tolstoy", "Dostoevsky"]
AUTHOR_STYLES = {
    "Tolstoy": np.full(8, 14.0, dtype=np.float32),
    "Dostoevsky": np.full(8, 16.0, dtype=np.float32),
}


class _Counter:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, normalize):
        texts = list(texts)
        self.calls.append(texts)
        return [_vector_for(text) for text in texts]

    def source_encodes(self):
        return sum(1 for call in self.calls for text in call
                   if text in SOURCES)


def _vector_for(text):
    # Distinct, non-parallel vectors per text so away/towards stay finite.
    seed = int.from_bytes(text.encode("utf-8")[:4].ljust(4, b"\0"),
                          "little")
    rng = np.random.default_rng(seed)
    return (rng.normal(size=8).astype(np.float32) * 5.0 + 14.0)


def _make_tst_func(n_versions):
    def fake(texts, target_style):
        if n_versions == 1:
            return [STYLED[SOURCES.index(t)] for t in texts]
        return [[STYLED[SOURCES.index(t)], STYLED_V1[SOURCES.index(t)]]
                for t in texts]
    return fake


def _install_counter(monkeypatch):
    import tst_utils.eval.performance.source_cache as source_cache_module
    import tst_utils.eval.performance.scoring as scoring_module
    counter = _Counter()
    monkeypatch.setattr(source_cache_module, "calc_style_embeddings", counter)
    monkeypatch.setattr(scoring_module, "calc_style_embeddings", counter)
    return counter


def _run(df, monkeypatch, n_versions, **execute_kwargs):
    counter = _install_counter(monkeypatch)
    pm = TstPerformanceMetrics(
        test_df=df,
        tst_func=_make_tst_func(n_versions),
        target_styles=TARGETS,
        tst_model="guard-e2e",
        author_styles=AUTHOR_STYLES,
        verbose=False,
    )
    pm.execute(**execute_kwargs)
    return pm, counter


def _base_df():
    return pd.DataFrame({"text": SOURCES, "author": ["News"] * len(SOURCES)})


@gpu_only
@pytest.mark.parametrize("n_versions", [1, 2])
def test_no_inline_column_no_warning_one_encode(monkeypatch, n_versions):
    import warnings
    df = _base_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error", StyleProvenanceWarning)
        _pm, counter = _run(df, monkeypatch, n_versions)
    assert counter.source_encodes() == len(SOURCES)


@gpu_only
@pytest.mark.parametrize("n_versions", [1, 2])
def test_inline_column_warns_and_reencodes(monkeypatch, n_versions):
    df = _base_df()
    df["text_style_emb"] = [np.zeros(8, dtype=np.float32)] * len(SOURCES)
    with pytest.warns(StyleProvenanceWarning, match="text_style_emb"):
        _pm, counter = _run(df, monkeypatch, n_versions)
    assert counter.source_encodes() == len(SOURCES)


@gpu_only
@pytest.mark.parametrize("n_versions", [1, 2])
def test_side_file_zero_source_encodes(monkeypatch, tmp_path, n_versions):
    import warnings
    from tst_utils.eval.data.style_store import (
        save_style_embeddings, measure_normalization,
    )
    from tst_utils.eval.metrics.style import EncodedStyle
    from tst_utils.eval.style_encoder_registry import get_encoder
    embeddings = 15.0 + (np.arange(len(SOURCES) * 8, dtype=np.float32)
                         .reshape(len(SOURCES), 8) % 7)
    label = {"encoder_key": "base_v1",
             "file_sha256": dict(get_encoder("base_v1")["file_sha256"]),
             "dim": 8, "n_rows": len(SOURCES)}
    label.update(measure_normalization(embeddings))
    path = str(tmp_path / "side.parquet")
    save_style_embeddings(SOURCES, EncodedStyle(embeddings=embeddings,
                                                label=label), path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StyleProvenanceWarning)
        _pm, counter = _run(_base_df(), monkeypatch, n_versions,
                            source_style_path=path)
    assert counter.source_encodes() == 0
