"""End-to-end guard test: TstPerformanceMetrics.execute with a canned tst_func.

1D (one output per input) and 2D (two outputs per input). The style
encoder's load seam (`style.load_style_encoder`, which every encode of
`encode_style` passes) is monkeypatched to a counter that returns fixed
vectors; every case asserts that it saw the styled-text encodes, so a zero
source count cannot pass vacuously (task 5.2). The other scorers (perplexity, BERTScore, LaBSE) run for real, so this file
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
# The fixtures name their encoder explicitly (task 5.2): a declared base_v1
# target source and an explicit base_v1 evaluation encoder.
ENCODER = "base_v1"


def _author_styles():
    from tst_utils.eval.data.style_store import LabelledStyleVectors
    return LabelledStyleVectors.declare(
        [np.full(8, 14.0, dtype=np.float32) + np.arange(8, dtype=np.float32),
         np.full(8, 16.0, dtype=np.float32) - np.arange(8, dtype=np.float32)],
        ENCODER, "guard-e2e probe", keys=TARGETS)


class _Counter:
    """A fake model on the one encode seam, `style.load_style_encoder`."""

    def __init__(self):
        self.calls = []

    def load(self, key):
        counter = self

        class Model:
            def encode(self, texts, show_progress_bar=False,
                       normalize_embeddings=False):
                texts = list(texts)
                counter.calls.append((key, texts))
                return np.stack([_vector_for(text) for text in texts])
        return Model()

    def source_encodes(self):
        return sum(1 for _key, call in self.calls for text in call
                   if text in SOURCES)

    def styled_encodes(self):
        styled = set(STYLED) | set(STYLED_V1)
        return sum(1 for _key, call in self.calls for text in call
                   if text in styled)


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
    import tst_utils.eval.metrics.style as style_module
    counter = _Counter()
    monkeypatch.setattr(style_module, "load_style_encoder", counter.load)
    return counter


def _run(df, monkeypatch, n_versions, **execute_kwargs):
    counter = _install_counter(monkeypatch)
    pm = TstPerformanceMetrics(
        test_df=df,
        tst_func=_make_tst_func(n_versions),
        target_styles=TARGETS,
        tst_model="guard-e2e",
        author_styles=_author_styles(),
        verbose=False,
        evaluation_encoder=ENCODER,
    )
    pm.execute(**execute_kwargs)
    # the counter sits on the encode route: every case sees the styled texts
    assert counter.styled_encodes() > 0
    assert {key for key, _ in counter.calls} == {ENCODER}
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
        save_style_embeddings, measure_normalization, style_text_key,
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
    save_style_embeddings(SOURCES, EncodedStyle(
        embeddings=embeddings, label=label,
        text_keys=tuple(style_text_key(t) for t in SOURCES)), path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StyleProvenanceWarning)
        _pm, counter = _run(_base_df(), monkeypatch, n_versions,
                            source_style_path=path)
    assert counter.source_encodes() == 0
