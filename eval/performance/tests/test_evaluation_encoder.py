"""Task 5.2, step 4: the evaluation encoder, the key-selected emb path and the
instance-owned source store of `TstPerformanceMetrics`.

CPU only, in the VM: the style encoder is the recording fake of `conftest.py`
(no snapshot is fetched, no model is loaded), and perplexity, BERTScore and
LaBSE are constant stubs. The pin source is patched through the accessor.
"""
import numpy as np
import pandas as pd
import pytest

from tst_utils.eval.data import load as load_module
from tst_utils.eval.data.style_store import (
    LabelledStyleVectors,
    StyleLabelError,
    StyleProvenanceWarning,
    measure_normalization,
    save_style_embeddings,
    style_text_key,
)
from tst_utils.eval.metrics import style as style_module
from tst_utils.eval.metrics.style import EncodedStyle, away, towards, unit_view
from tst_utils.eval.performance.core import TstPerformanceMetrics
from tst_utils.eval.style_encoder_registry import get_encoder

from conftest import DIM, fake_vector

SOURCES = ["первый источник", "второй источник", "третий источник"]
TEXT_TARGETS = ["Tolstoy", "Dostoevsky"]


def _styled(text, version, style):
    return f"{text} -> {style} v{version}"


def _text_func(n_versions=2):
    calls = []

    def tst_func(texts, target_style):
        calls.append((list(texts), target_style))
        return [[_styled(t, v, target_style) for v in range(n_versions)]
                for t in texts]
    tst_func.calls = calls
    return tst_func


def _emb_func(n_versions=2, require=None):
    calls = []

    def tst_func(texts, target_style_embeddings):
        selection = target_style_embeddings
        if require is not None:
            # a stub model conditioned on `require`
            assert selection.encoder_key == require
        calls.append(selection)
        return [[_styled(t, v, k) for v in range(n_versions)]
                for t, k in zip(texts, selection.row_keys)]
    tst_func.calls = calls
    return tst_func


def _author_targets(key, source="probe centroids"):
    names = TEXT_TARGETS + ["News"]
    return LabelledStyleVectors.declare(
        [fake_vector(key, f"centroid {n}") for n in names], key, source,
        keys=names)


def _text_df():
    return pd.DataFrame({"text": SOURCES, "author": ["News"] * len(SOURCES)})


def _text_pm(evaluation_encoder=None, n_versions=2, **kwargs):
    key = evaluation_encoder or "base_v2"
    return TstPerformanceMetrics(
        test_df=_text_df(), tst_func=_text_func(n_versions),
        target_styles=TEXT_TARGETS, tst_model="probe",
        author_styles=kwargs.pop("author_styles", _author_targets(key)),
        evaluation_encoder=evaluation_encoder, **kwargs)


def _side_file(tmp_path, key, texts=SOURCES, offset=1.0):
    embeddings = np.stack([fake_vector(key, t) + offset for t in texts])
    label = {"encoder_key": key,
             "file_sha256": dict(get_encoder(key)["file_sha256"]),
             "dim": DIM, "n_rows": len(texts)}
    label.update(measure_normalization(embeddings))
    path = str(tmp_path / f"side_{key}.parquet")
    save_style_embeddings(texts, EncodedStyle(
        embeddings=embeddings, label=label,
        text_keys=tuple(style_text_key(t) for t in texts)), path)
    stored = embeddings.astype(np.float16).astype(np.float32)  # fp16 on disk
    return path, {t: stored[i] for i, t in enumerate(texts)}


# ---- requirement 3: v1 conditioning, v2 scoring ------------------------------

def test_named_target_run_conditions_in_v1_and_scores_in_v2(
        pin, fake_encoder, stub_scorers):
    pin("base_v2")
    conditioning = load_module.load_labelled_author_styles(
        expect_encoder="base_v1", path=load_module.AUTHOR_STYLES_BASE_V1_PATH)
    used = []

    def tst_func(texts, target_style):
        # the generator side: a base_v1 checkpoint fed from the v1 copy
        used.append(conditioning.select([target_style] * len(texts)).encoder_key)
        return [_styled(t, 0, target_style) for t in texts]

    pm = TstPerformanceMetrics(
        test_df=_text_df(), tst_func=tst_func, target_styles=TEXT_TARGETS,
        tst_model="v1-checkpoint", author_styles=_author_targets("base_v2"))
    pm.execute()
    assert set(used) == {"base_v1"}
    assert fake_encoder.keys() and set(fake_encoder.keys()) == {"base_v2"}
    assert set(pm.tst_results["evaluation_encoder"]) == {"base_v2"}
    assert pm.final_performance()["evaluation_encoder"] == "base_v2"
    # the v1 conditioning source is refused as the scoring source
    with pytest.raises(StyleLabelError, match="base_v1"):
        TstPerformanceMetrics(
            test_df=_text_df(), tst_func=tst_func, target_styles=TEXT_TARGETS,
            tst_model="x", author_styles=conditioning)


# ---- requirement 3: an explicit base_v1 arbitrary-target run ----------------

def _v1_targets(kind, fake_encoder):
    keys = ["writer_a", "writer_b"]
    if kind == "encode":
        encoded = style_module.encode_style(["образец a", "образец b"],
                                            "base_v1", normalize=False)
        return LabelledStyleVectors.from_encoded(encoded, keys=keys)
    return LabelledStyleVectors.declare(
        [fake_vector("base_v1", "declared a"), fake_vector("base_v1", "declared b")],
        "base_v1", "operator vectors", keys=keys)


@pytest.mark.parametrize("kind", ["encode", "declare"])
def test_explicit_v1_arbitrary_target_run(pin, fake_encoder, stub_scorers,
                                          tmp_path, kind):
    pin("base_v2")
    targets = _v1_targets(kind, fake_encoder)
    df = pd.DataFrame({"text": SOURCES, "author": ["News"] * 3,
                       "target_style_key": ["writer_b", "writer_a", "writer_b"],
                       "target_style_desc": ["probe"] * 3})
    tst_func = _emb_func(n_versions=2, require="base_v1")
    pm = TstPerformanceMetrics(
        test_df=df, tst_func=tst_func, target_styles=None, tst_model="v1",
        author_styles=None, evaluation_encoder="base_v1",
        target_style_source=targets)
    pm.execute()
    assert set(fake_encoder.keys()) == {"base_v1"}
    assert len(tst_func.calls) == 1
    results = pm.tst_results
    assert len(results) == 6
    assert set(results["target_provenance"]) == {kind}
    assert set(results["target_declared"]) == {kind == "declare"}
    assert set(results["evaluation_encoder"]) == {"base_v1"}
    path = str(tmp_path / "results.parquet")
    results.drop(columns=["text_style_emb", "styled_text_style_emb"]).to_parquet(path)
    back = pd.read_parquet(path)
    assert list(back["target_provenance"]) == list(results["target_provenance"])
    assert list(back["target_declared"]) == list(results["target_declared"])
    assert list(back["target_source"]) == list(results["target_source"])
    assert set(pm.best_tst_results["target_provenance"]) == {kind}
    assert pm.final_performance()["evaluation_encoder"] == "base_v1"


def test_wrong_target_encoder_raises_before_the_tst_callback(pin, fake_encoder):
    pin("base_v2")
    tst_func = _emb_func()
    df = pd.DataFrame({"text": SOURCES[:1], "author": ["News"],
                       "target_style_key": ["a"], "target_style_desc": ["x"]})
    with pytest.raises(StyleLabelError, match="base_v2"):
        TstPerformanceMetrics(
            test_df=df, tst_func=tst_func, target_styles=None, tst_model="x",
            author_styles=None, evaluation_encoder="base_v1",
            target_style_source=LabelledStyleVectors.declare(
                [np.ones(DIM, dtype=np.float32)], "base_v2", "v2", keys=["a"]))
    assert tst_func.calls == [] and fake_encoder.calls == []


def test_unknown_or_missing_keys_raise_before_any_encode_or_tst_call(
        pin, fake_encoder):
    pin("base_v2")
    for keys, error in ((["a", "zzz"], KeyError), (["a", None], ValueError)):
        tst_func = _emb_func()
        df = pd.DataFrame({"text": SOURCES[:2], "author": ["News"] * 2,
                           "target_style_key": keys,
                           "target_style_desc": ["x", "y"]})
        pm = TstPerformanceMetrics(
            test_df=df, tst_func=tst_func, target_styles=None, tst_model="x",
            author_styles=None, target_style_source=LabelledStyleVectors.declare(
                [np.ones(DIM, dtype=np.float32)], "base_v2", "v2", keys=["a"]))
        with pytest.raises(error):
            pm.produce_tst_results()
        assert tst_func.calls == [] and fake_encoder.calls == []


# ---- the default readers and an explicit override ----------------------------

def test_default_and_override_are_resolved_once(pin):
    pin("base_v2")
    default = _text_pm()
    override = _text_pm(evaluation_encoder="base_v1")
    pin("base_v1")
    assert default.evaluation_encoder == "base_v2"
    assert override.evaluation_encoder == "base_v1"
    assert _text_pm(evaluation_encoder="base_v1").evaluation_encoder == "base_v1"
    with pytest.raises(AttributeError):
        default.evaluation_encoder = "base_v1"


def test_final_performance_returns_the_instance_encoder(pin, fake_encoder,
                                                        stub_scorers):
    pin("base_v2")
    pm = _text_pm(evaluation_encoder="base_v1")
    pm.execute()
    pin("base_v2")
    result = pm.final_performance()
    assert result["evaluation_encoder"] == "base_v1"
    assert set(fake_encoder.keys()) == {"base_v1"}


def test_bare_dicts_and_inline_targets_raise_off_base_v1(pin, fake_encoder):
    pin("base_v2")
    bare = {"Tolstoy": np.ones(DIM, dtype=np.float32)}
    with pytest.raises(StyleLabelError, match="bare dict"):
        _text_pm(author_styles=bare)
    df = pd.DataFrame({"text": SOURCES[:1], "author": ["News"],
                       "target_style_emb": [np.ones(DIM, dtype=np.float32)],
                       "target_style_desc": ["x"]})
    tst_func = _emb_func()
    pm = TstPerformanceMetrics(test_df=df, tst_func=tst_func, target_styles=None,
                               tst_model="x", author_styles=None)
    with pytest.raises(StyleLabelError, match="target_style_emb"):
        pm.produce_tst_results()
    # an explicit base_v1 run does not reopen the legacy intake
    with pytest.raises(StyleLabelError, match="bare dict"):
        _text_pm(evaluation_encoder="base_v1", author_styles=bare)
    assert tst_func.calls == [] and fake_encoder.calls == []


def test_legacy_inline_target_is_a_declared_v1_source_while_pin_is_v1(
        pin, fake_encoder, stub_scorers):
    pin("base_v1")
    vectors = [fake_vector("base_v1", f"inline {i}") for i in range(3)]
    df = pd.DataFrame({"text": SOURCES, "author": ["News"] * 3,
                       "target_style_emb": vectors,
                       "target_style_desc": ["x"] * 3})
    tst_func = _emb_func(n_versions=1, require="base_v1")
    pm = TstPerformanceMetrics(test_df=df, tst_func=tst_func, target_styles=None,
                               tst_model="x", author_styles=None)
    with pytest.warns(StyleProvenanceWarning, match="target_style_emb"):
        pm.execute()
    selection = tst_func.calls[0]
    assert selection.source.is_declared
    assert np.array_equal(selection.vectors(), np.stack(vectors))
    assert set(pm.tst_results["target_declared"]) == {True}


# ---- repeated and noncontiguous keys ---------------------------------------

def test_repeated_noncontiguous_keys_through_expansion_and_scoring(
        pin, fake_encoder, stub_scorers, tmp_path):
    pin("base_v2")
    source = LabelledStyleVectors.declare(
        [fake_vector("base_v2", k) for k in "abc"], "base_v2", "probe",
        keys=["a", "b", "c"])
    texts = SOURCES + ["четвёртый источник"]
    row_keys = ["c", "a", "c", "b"]
    df = pd.DataFrame({"text": texts, "author": ["News"] * 4,
                       "target_style_key": row_keys,
                       "target_style_desc": ["x"] * 4})
    tst_func = _emb_func(n_versions=2)
    pm = TstPerformanceMetrics(test_df=df, tst_func=tst_func, target_styles=None,
                               tst_model="x", author_styles=None,
                               target_style_source=source)
    pm.execute()
    assert tst_func.calls[0].row_keys == tuple(row_keys)
    results = pm.tst_results
    by_example = results.groupby("example_number")["target_style_key"].agg(set)
    assert [by_example[i] for i in range(4)] == [{k} for k in row_keys]
    for _, row in results.iterrows():
        src = fake_vector("base_v2", row.text)
        cur = fake_vector("base_v2", row.styled_text)
        tgt = unit_view(source.vector(row.target_style_key))
        assert row.away == pytest.approx(away(src, cur, tgt))
        assert row.towards == pytest.approx(towards(src, cur, tgt))
    path = str(tmp_path / "r.parquet")
    results[["example_number", "version_number", "target_style_key"]].to_parquet(path)
    assert list(pd.read_parquet(path)["target_style_key"]) == list(
        results["target_style_key"])


def test_scoring_unit_view_is_derived_and_never_written_back(
        pin, fake_encoder, stub_scorers):
    pin("base_v2")
    targets = _author_targets("base_v2")
    before = targets.matrix().copy()
    pm = _text_pm(author_styles=targets)
    pm.execute()
    assert np.array_equal(targets.matrix(), before)
    assert np.all(targets.norms() > 10.0)


# ---- requirement 11: every public entry validates ----------------------------

ENTRIES = ["add_source_ppx_and_emb", "produce_tst_results",
           "produce_tst_results_with_text_target_styles",
           "produce_tst_results_with_target_style_emb", "compute_scores"]


def _entry_pm(entry):
    if entry == "produce_tst_results_with_target_style_emb":
        df = pd.DataFrame({"text": SOURCES, "author": ["News"] * 3,
                           "target_style_key": ["Tolstoy"] * 3,
                           "target_style_desc": ["x"] * 3})
        return TstPerformanceMetrics(
            test_df=df, tst_func=_emb_func(), target_styles=None,
            tst_model="x", author_styles=None,
            target_style_source=_author_targets("base_v2"))
    return _text_pm()


def _call(pm, entry):
    if entry == "compute_scores":
        pm.produce_tst_results()
    getattr(pm, entry)()


def _stale(pm, entry):
    """Replace the source column the entry consumes with stale values."""
    frame = "tst_results" if entry == "compute_scores" else "test_df"
    df = getattr(pm, frame).copy()
    df["text_style_emb"] = [fake_vector("base_v1", t) for t in df.text]
    setattr(pm, frame, df)


@pytest.mark.parametrize("entry", ENTRIES)
def test_each_entry_discards_and_reencodes_a_stale_inline_source(
        pin, fake_encoder, stub_scorers, entry):
    pin("base_v2")
    pm = _entry_pm(entry)
    if entry == "compute_scores":
        pm.produce_tst_results()
    _stale(pm, entry)
    fake_encoder.calls.clear()
    with pytest.warns(StyleProvenanceWarning, match="text_style_emb"):
        getattr(pm, entry)()
    assert fake_encoder.count(SOURCES) == len(SOURCES)
    frame = pm.tst_results if entry == "compute_scores" else pm.test_df
    for _, row in frame.iterrows():
        assert np.array_equal(row.text_style_emb, fake_vector("base_v2", row.text))


@pytest.mark.parametrize("entry", ENTRIES)
def test_each_entry_raises_on_a_mismatched_side_file(
        pin, fake_encoder, stub_scorers, tmp_path, entry):
    pin("base_v2")
    path, _ = _side_file(tmp_path, "base_v1")
    pm = _entry_pm(entry)
    if entry == "compute_scores":
        pm.produce_tst_results()  # a warm store, then the side file
    pm.source_style_path = path
    with pytest.raises(StyleLabelError, match="base_v1"):
        getattr(pm, entry)()


# ---- requirement 11: the source store is owned by the instance -------------

def _warm(route, tmp_path):
    """A text-path instance after the production entry, its store filled by
    its own encode or by a checked side-file load. Returns (pm, expected)."""
    if route == "side":
        path, values = _side_file(tmp_path, "base_v2")
        pm = _text_pm(source_style_path=path)
    else:
        pm, values = _text_pm(), {t: fake_vector("base_v2", t) for t in SOURCES}
    pm.produce_tst_results()
    return pm, values


def _assert_scored_with(pm, values):
    for _, row in pm.tst_results.iterrows():
        tgt = unit_view(pm.author_styles.vector(row.target_style))
        cur = fake_vector("base_v2", row.styled_text)
        assert row.away == pytest.approx(away(values[row.text], cur, tgt))


@pytest.mark.parametrize("route", ["encode", "side"])
def test_positive_control_warm_store_zero_source_encodes(
        pin, fake_encoder, stub_scorers, tmp_path, route):
    import warnings
    pin("base_v2")
    pm, values = _warm(route, tmp_path)
    # repeated texts across target styles and versions
    assert pm.tst_results.text.duplicated().any()
    fake_encoder.calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", StyleProvenanceWarning)
        pm.produce_tst_results()
        pm.compute_scores()
    assert fake_encoder.count(SOURCES) == 0
    assert len(fake_encoder.encoded_texts()) > 0  # the styled-text encodes
    _assert_scored_with(pm, values)


@pytest.mark.parametrize("route", ["encode", "side"])
@pytest.mark.parametrize("case", ["a_test_df_v1_column", "b_result_cell",
                                  "c_in_place_mutation", "d_unseen_text",
                                  "e_one_of_repeated_rows"])
def test_source_store_ownership(pin, fake_encoder, stub_scorers, tmp_path,
                                route, case):
    pin("base_v2")
    pm, values = _warm(route, tmp_path)
    store_before = {k: v.copy() for k, v in pm._source_store.items()}
    fake_encoder.calls.clear()
    if case == "a_test_df_v1_column":
        df = pm.test_df.copy()
        df["text_style_emb"] = [fake_vector("base_v1", t) for t in df.text]
        pm.test_df = df
        affected, entry = set(SOURCES), "produce_tst_results"
    elif case == "b_result_cell":
        test_df_before = pm.test_df.copy()
        pm.tst_results.at[0, "text_style_emb"] = np.zeros(DIM, dtype=np.float32)
        affected, entry = {pm.tst_results.at[0, "text"]}, "compute_scores"
    elif case == "c_in_place_mutation":
        exposed = pm.tst_results.at[1, "text_style_emb"]
        exposed[:] = -1.0
        affected, entry = {pm.tst_results.at[1, "text"]}, "compute_scores"
    elif case == "d_unseen_text":
        unseen = "текст, которого хранилище не видело"
        pm.tst_results.loc[pm.tst_results.text == SOURCES[2], "text"] = unseen
        values = dict(values, **{unseen: fake_vector("base_v2", unseen)})
        affected, entry = {unseen}, "compute_scores"
        if route == "side":
            # a side file that lacks the text is a hard error, not a guess
            with pytest.raises(StyleLabelError, match="no row"):
                pm.compute_scores()
            return
    else:
        repeated = pm.tst_results.index[pm.tst_results.text == SOURCES[0]]
        assert len(repeated) > 1
        pm.tst_results.at[repeated[0], "text_style_emb"] = np.full(
            DIM, 3.0, dtype=np.float32)
        affected, entry = {SOURCES[0]}, "compute_scores"
    with pytest.warns(StyleProvenanceWarning, match="text_style_emb"):
        getattr(pm, entry)()
    expected_encodes = 0 if route == "side" else len(affected)
    if case == "d_unseen_text":
        expected_encodes = 1
    assert fake_encoder.count(set(SOURCES) | affected) == expected_encodes
    assert set(fake_encoder.encoded_texts()) - {
        t for t in fake_encoder.encoded_texts() if " -> " in t} <= affected
    # no public mutation altered an authoritative entry
    for key, vector in store_before.items():
        assert np.array_equal(pm._source_store[key], vector)
        assert not pm._source_store[key].flags.writeable
    if case == "b_result_cell":
        pd.testing.assert_frame_equal(pm.test_df, test_df_before)
    if entry == "compute_scores":
        _assert_scored_with(pm, values)
    else:
        for _, row in pm.test_df.iterrows():
            assert np.array_equal(row.text_style_emb, values[row.text])


# ---- the calc_style_embeddings warning (host ruling P11) ---------------------

def test_calc_style_embeddings_warns_once_off_base_v1(pin, fake_encoder,
                                                      monkeypatch):
    import warnings
    monkeypatch.setitem(style_module._historical_pin_warning, "shown", False)
    pin("base_v1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        style_module.calc_style_embeddings(["x"], normalize=False)
        style_module.calc_style_embeddings(["y"], normalize=False)
    assert not [w for w in caught if "calc_style_embeddings" in str(w.message)]
    pin("base_v2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        style_module.calc_style_embeddings(["x"], normalize=False)
        style_module.calc_style_embeddings(["y"], normalize=False)
    pin_warnings = [w for w in caught if "calc_style_embeddings" in str(w.message)]
    assert len(pin_warnings) == 1
    assert "base_v2" in str(pin_warnings[0].message)
    assert fake_encoder.keys() == ["base_v1", "base_v1", "base_v2", "base_v2"]
