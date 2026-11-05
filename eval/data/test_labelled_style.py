"""Task 5.2, step 3: labelled style vectors, selections, the checked loader
and the call-time pin accessor. CPU only; no model is loaded."""
import numpy as np
import pandas as pd
import pytest

from tst_utils.eval import model_names
from tst_utils.eval.data import load as load_module
from tst_utils.eval.data import style_store as store_module
from tst_utils.eval.data.style_store import (
    LabelledStyleVectors,
    StyleLabelError,
    StyleSelection,
    build_centroids,
    measure_normalization,
    style_text_key,
)
from tst_utils.eval.style_encoder_registry import get_encoder


def _encoded(n, key="base_v1", scale=15.0):
    from tst_utils.eval.metrics.style import EncodedStyle
    embeddings = scale + (np.arange(n * 4, dtype=np.float32).reshape(n, 4) % 5)
    texts = ["text %d" % i for i in range(n)]
    label = {"encoder_key": key,
             "file_sha256": dict(get_encoder(key)["file_sha256"]),
             "dim": 4, "n_rows": n}
    label.update(measure_normalization(embeddings))
    return EncodedStyle(embeddings=embeddings, label=label,
                        text_keys=tuple(style_text_key(t) for t in texts))


def _declared():
    vectors = [np.full(4, 1.0, dtype=np.float32), np.full(4, 2.0, dtype=np.float32),
               np.full(4, 3.0, dtype=np.float32)]
    return LabelledStyleVectors.declare(vectors, "base_v1", "probe vectors",
                                        keys=["a", "b", "c"]), vectors


# ---- the swap counterexample is unexpressible ------------------------------

def test_direct_construction_is_refused():
    with pytest.raises(TypeError, match="load_labelled_centroids"):
        LabelledStyleVectors(keys=["a"], vectors=[np.ones(4)],
                             encoder_key="base_v1", provenance="loader",
                             source="x", label={})


def test_object_is_immutable():
    styles, _ = _declared()
    with pytest.raises(AttributeError):
        styles._encoder_key = "base_v2"
    with pytest.raises(AttributeError):
        styles._matrix = np.zeros((3, 4))
    with pytest.raises(AttributeError):
        del styles._keys


def test_swapping_vectors_after_construction_is_unexpressible():
    """The planning-party counterexample: relabel or swap the arrays behind
    a checked label. The caller's arrays are copies, the stored ones are
    read-only, and no public setter exists."""
    styles, vectors = _declared()
    vectors[0][:] = 99.0
    vectors.reverse()
    assert np.array_equal(styles.vector("a"), np.full(4, 1.0))
    view = styles.vector("a")
    with pytest.raises(ValueError):
        view[:] = 5.0
    with pytest.raises(ValueError):
        styles.matrix()[1, 0] = 7.0
    assert np.array_equal(styles.vector("b"), np.full(4, 2.0))


def test_label_is_a_copy():
    styles, _ = _declared()
    label = styles.label
    label["encoder_key"] = "base_v2"
    assert styles.label["encoder_key"] == "base_v1"


# ---- constructors ------------------------------------------------------------

def test_keys_must_be_unique_and_not_reserved():
    with pytest.raises(ValueError, match="unique"):
        LabelledStyleVectors.declare([np.ones(4), np.ones(4)], "base_v1", "x",
                                     keys=["a", "a"])
    with pytest.raises(ValueError, match="__"):
        LabelledStyleVectors.declare([np.ones(4)], "base_v1", "x",
                                     keys=["__style_label__"])
    with pytest.raises(ValueError, match="keys for"):
        LabelledStyleVectors.declare([np.ones(4)], "base_v1", "x",
                                     keys=["a", "b"])


def test_unregistered_encoder_raises():
    with pytest.raises(Exception):
        LabelledStyleVectors.declare([np.ones(4)], "base_v9", "x", keys=["a"])


def test_declaration_is_reported_as_a_declaration():
    styles, _ = _declared()
    assert styles.provenance == "declare"
    assert styles.is_declared
    assert styles.describe() == {"encoder_key": "base_v1",
                                 "provenance": "declare",
                                 "source": "probe vectors", "n_keys": 3,
                                 "declared": True}


def test_from_encoded_needs_encoded_style_and_keys():
    encoded = _encoded(3)
    styles = LabelledStyleVectors.from_encoded(encoded, keys=["x", "y", "z"])
    assert styles.provenance == "encode"
    assert styles.encoder_key == "base_v1"
    assert not styles.is_declared
    assert np.array_equal(styles.vector("y"), encoded.embeddings[1])
    encoded.embeddings[1, 0] = -1.0
    assert styles.vector("y")[0] != -1.0
    with pytest.raises(TypeError, match="EncodedStyle"):
        LabelledStyleVectors.from_encoded(
            {"embeddings": encoded.embeddings, "label": encoded.label},
            keys=["x", "y", "z"])
    with pytest.raises(ValueError, match="keys for"):
        LabelledStyleVectors.from_encoded(encoded, keys=["x"])


def test_from_build_takes_the_builder_label(tmp_path):
    encoded = _encoded(4, key="base_v2")
    built = build_centroids(encoded, ["a", "a", "b", "b"], renormalize=False,
                            path=str(tmp_path / "c.npz"))
    styles = LabelledStyleVectors.from_build(built, source="probe build")
    assert styles.encoder_key == "base_v2"
    assert styles.provenance == "build"
    assert styles.keys == ("a", "b")


# ---- selections --------------------------------------------------------------

def test_selection_keeps_repeats_and_order_and_copies_the_key_list():
    styles, _ = _declared()
    row_keys = ["c", "a", "c", "b", "a"]
    selection = styles.select(row_keys)
    row_keys[0] = "b"
    row_keys.append("a")
    assert selection.row_keys == ("c", "a", "c", "b", "a")
    assert selection.encoder_key == "base_v1"
    gathered = selection.vectors()
    assert gathered[:, 0].tolist() == [3.0, 1.0, 3.0, 2.0, 1.0]
    gathered[0, 0] = 42.0
    assert styles.vector("c")[0] == 3.0


def test_selection_noncontiguous_subset():
    styles, _ = _declared()
    selection = styles.select(["c", "a"])
    assert selection.vectors()[:, 0].tolist() == [3.0, 1.0]
    assert len(selection) == 2


def test_selection_rejects_unknown_keys_bare_dicts_and_strings():
    styles, _ = _declared()
    with pytest.raises(KeyError, match="zzz"):
        styles.select(["a", "zzz"])
    with pytest.raises(TypeError, match="LabelledStyleVectors"):
        StyleSelection({"a": np.ones(4)}, ["a"])
    with pytest.raises(TypeError, match="not a str"):
        styles.select("abc")
    with pytest.raises(AttributeError):
        styles.select(["a"])._row_keys = ("b",)


# ---- the checked loader ----------------------------------------------------

def test_checked_loader_reads_the_v1_author_styles():
    styles = load_module.load_labelled_author_styles(
        expect_encoder="base_v1", path=load_module.AUTHOR_STYLES_BASE_V1_PATH)
    assert styles.provenance == "loader"
    assert styles.encoder_key == "base_v1"
    assert set(styles.keys) == {"Bible", "Chekhov", "Dostoevsky", "Tolstoy",
                                "News"}
    with np.load(load_module.AUTHOR_STYLES_BASE_V1_PATH) as raw:
        assert np.array_equal(styles.vector("Tolstoy"), raw["Tolstoy"])
    assert 14.0 < float(styles.norms().min())


def test_checked_loader_rejects_a_wrong_encoder_and_an_unlabelled_file(tmp_path):
    with pytest.raises(StyleLabelError, match="base_v1"):
        load_module.load_labelled_author_styles(
            expect_encoder="base_v2", path=load_module.AUTHOR_STYLES_BASE_V1_PATH)
    path = str(tmp_path / "u.npz")
    np.savez(path, a=np.ones(768, dtype=np.float32))
    with pytest.raises(StyleLabelError, match="unlabelled"):
        load_module.load_labelled_centroids(path, expect_encoder="base_v1")


def test_checked_loader_round_trips_a_build(tmp_path):
    path = str(tmp_path / "c.npz")
    centroids, _ = build_centroids(_encoded(4, key="base_v2"),
                                   ["a", "b", "a", "b"], renormalize=True,
                                   path=path)
    styles = load_module.load_labelled_centroids(path, expect_encoder="base_v2")
    assert np.array_equal(styles.vector("b"), centroids["b"])


# ---- one pin source reaches every reader ------------------------------------

def test_patch_of_the_pin_source_reaches_every_reader(monkeypatch, tmp_path):
    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", "base_v2")
    assert model_names.style_encoder_key() == "base_v2"
    # the v1 copy is labelled base_v1: every default reader refuses it
    v1_path = load_module.AUTHOR_STYLES_BASE_V1_PATH
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_module.load_centroids_npz(v1_path, renormalize=False)
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_module.load_labelled_author_styles(path=v1_path)
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_module.load_labelled_centroids(v1_path)
    # side files: a v1 side file is refused by the default
    from tst_utils.eval.data.style_store import (join_style_embeddings,
                                                 load_style_embeddings,
                                                 save_style_embeddings)
    texts = ["text 0", "text 1"]
    side = str(tmp_path / "side.parquet")
    save_style_embeddings(texts, _encoded(2), side)
    with pytest.raises(StyleLabelError, match="base_v2"):
        load_style_embeddings(side)
    with pytest.raises(StyleLabelError, match="base_v2"):
        join_style_embeddings(pd.DataFrame({"text": texts}), side, "text",
                              "text_style_emb")
    # the inline target column raises once the pin is not base_v1
    with pytest.raises(StyleLabelError, match="base_v2"):
        store_module.guard_inline_style_columns(
            pd.DataFrame({"text": ["a"],
                          "target_style_emb": [np.ones(4, dtype=np.float32)]}),
            entry_point="probe")
    # calc_style_embeddings loads the patched pin (a spy, no model load)
    from tst_utils.eval.metrics import style as style_module
    requested = []

    class FakeModel:
        def encode(self, texts, show_progress_bar, normalize_embeddings):
            return np.ones((len(texts), 4), dtype=np.float32)

    def spy(key):
        requested.append(key)
        return FakeModel()

    monkeypatch.setattr(style_module, "load_style_encoder", spy)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        style_module.calc_style_embeddings(["x"], normalize=False)
    assert requested == ["base_v2"]


def test_no_import_time_copy_of_the_pin_remains():
    import tst_utils.eval.metrics.style as style_module
    for module in (load_module, store_module, style_module):
        assert not hasattr(module, "STYLE_ENCODER_KEY"), module.__name__


def test_v1_copy_loads_as_base_v1():
    styles = load_module.load_labelled_author_styles(
        expect_encoder="base_v1", path=load_module.AUTHOR_STYLES_BASE_V1_PATH)
    assert styles.encoder_key == "base_v1"
    assert len(styles) == 5


def test_load_style_encoder_passes_fix_mistral_regex_false(monkeypatch):
    """The documented treatment: an explicit False at the tokenizer load,
    never True (a stub; no snapshot is fetched and no model is loaded)."""
    from tst_utils.eval.metrics import style as style_module
    seen = {}
    monkeypatch.setattr(style_module, "snapshot_download",
                        lambda hub_id, revision: "/nonexistent/snapshot")
    monkeypatch.setattr(style_module, "check_snapshot_hashes",
                        lambda *args: None)

    def fake_sentence_transformer(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(style_module, "SentenceTransformer",
                        fake_sentence_transformer)
    style_module.load_style_encoder("base_v1")
    assert seen["kwargs"] == {"tokenizer_kwargs": {"fix_mistral_regex": False}}
