"""Fakes for the CPU tests of `TstPerformanceMetrics` (task 5.2).

`fake_encoder` replaces the one model-load seam of every style encode,
`tst_utils.eval.metrics.style.load_style_encoder`, with a recording fake:
`encode_style` and `calc_style_embeddings` still run, the label still
comes from the registry, and no snapshot is fetched or loaded. Each call
records the encoder key and the texts, and the vectors depend on both, so
a v1 and a v2 vector of one text differ.

`stub_scorers` replaces the perplexity, BERTScore and LaBSE steps (they load
models) with constant columns; the style step runs for real on the fake
encoder.
"""
import hashlib

import numpy as np
import pytest

DIM = 8


def fake_vector(key, text):
    seed = int.from_bytes(hashlib.sha256(f"{key}|{text}".encode()).digest()[:8],
                          "little")
    rng = np.random.default_rng(seed)
    return (rng.normal(size=DIM) * 3.0 + 15.0).astype(np.float32)


class FakeEncoderRoute:
    def __init__(self):
        self.calls = []  # (encoder key, [texts])

    def load(self, key):
        route = self

        class Model:
            def encode(self, texts, show_progress_bar=False,
                       normalize_embeddings=False):
                texts = list(texts)
                route.calls.append((key, texts))
                out = np.stack([fake_vector(key, t) for t in texts])
                if normalize_embeddings:
                    out = out / np.linalg.norm(out, axis=1, keepdims=True)
                return out
        return Model()

    def keys(self):
        return [key for key, _ in self.calls]

    def encoded_texts(self, key=None):
        return [t for k, texts in self.calls for t in texts
                if key is None or k == key]

    def count(self, texts):
        texts = set(texts)
        return sum(1 for t in self.encoded_texts() if t in texts)


@pytest.fixture
def fake_encoder(monkeypatch):
    from tst_utils.eval.metrics import style as style_module
    route = FakeEncoderRoute()
    monkeypatch.setattr(style_module, "load_style_encoder", route.load)
    return route


@pytest.fixture
def stub_scorers(monkeypatch):
    from tst_utils.eval.performance import core, scoring

    def perplexity(df, log=None, batch_size=32):
        df["text_perplexity"] = 20.0
        df["styled_text_perplexity"] = 25.0

    def bert(df, log=None):
        df["bert_score"] = 0.8

    def labse(df, log=None):
        df["labse_score"] = 0.9

    monkeypatch.setattr(scoring, "add_perplexity_scores", perplexity)
    monkeypatch.setattr(scoring, "add_bert_score", bert)
    monkeypatch.setattr(scoring, "add_labse_score", labse)
    monkeypatch.setattr(core, "ensure_source_caches", lambda *a, **k: None)


@pytest.fixture
def pin(monkeypatch):
    from tst_utils.eval import model_names

    def set_pin(key):
        monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", key)
    return set_pin
