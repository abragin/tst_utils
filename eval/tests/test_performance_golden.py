"""End-to-end golden regression for `TstPerformanceMetrics` (task 2A.5).

Freezes the *current* behaviour of the full performance pipeline
(`execute()` + `compute_quality_scores()` + `compute_composite_v2()`) so the
2A.5 refactor (dedup 1D/2D reshaping, decompose `compute_scores`, consolidate
caching, package split) can be proven behaviour-preserving float-exact.

Design (decided 2026-06-06, see docs/tasks/2026-06-06-performance-py-refactor.md):

  - **Canned deterministic `tst_func`.** The refactor does NOT touch generation;
    `tst_func` is external. The real sampler (`GEN_OPTS_V3`, num_sequences=5) is
    not float-reproducible, which would make a golden meaningless. So a stub
    returns FIXED styled strings (`CANNED`), which still drives every 1D/2D
    reshaping branch and every (deterministic) scoring model.

  - **Four cases** = {text-target-style, target-style-emb} × {1D, 2D}, covering
    both `produce_tst_results_*` paths and both output shapes.

  - **Scorer config mirrors production eval** (`15 metrics exploration/step6_eval.py`):
    BatchedAligner ruRoberta-L8 rolling_max w9 + rubert-tiny2-L3 span4 fallback,
    GenderConsistencyScorer, EntityConsistencyScorer(pre_clean=False).

GPU-gated (needs rugpt3small / BERTScore / LaBSE / style ST / ruRoberta /
gender / entity). Run on tallin:

    pytest tst_utils/eval/tests/test_performance_golden.py

Known limit: the golden is blind to the entity pillar (measured 2026-10-06,
on tallin, task 2026-10-03-pin-scorer-model-revisions). The `entity_score`
column is 1.0 in all 54 frozen rows, because no canned styled text carries a
new entity. So a green golden is no evidence that the NER model or the entity
scorer is unchanged, and `score_v2` cannot show an entity change through the
golden either. The likely failure of a model revision change is a loss of NER
recall, which moves `entity_score` toward 1.0, where it already is. The other
9 columns have 3 to 24 distinct values. The fix: add one canned pair whose
styled side carries a target-author person name that is absent from the
source. It changes the baseline, so it needs its own task; no task owns it
(host, 2026-10-06).

Regenerate the frozen baseline (only when behaviour intentionally changes —
NOT as part of the behaviour-preserving refactor):

    python tst_utils/eval/tests/test_performance_golden.py --generate [--out-dir DIR] [--note TEXT]

`--out-dir` (task 5.2) writes `performance_golden.json` into DIR instead of
this folder, for a job container whose code tree is read-only.

Supply route (task 5.2): the harness reads the canonical author_styles
through the checked loader at the pin, as raw labelled vectors; the scoring
consumer derives the unit view. The emb cases select their target by
`target_style_key` from that same labelled source. The inline
`target_style_emb` rejection is its own pin-parameterized CPU test below.
"""

import gc
import json
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tst_utils.eval.performance import TstPerformanceMetrics
from tst_utils.eval.data.load import load_labelled_author_styles
from tst_utils.eval.data.style_store import StyleLabelError, StyleProvenanceWarning

HAS_CUDA = torch.cuda.is_available()
gpu_only = pytest.mark.skipif(
    not HAS_CUDA, reason="full eval pipeline — run on tallin GPU")

GOLDEN = os.path.join(os.path.dirname(__file__), "performance_golden.json")
TOL = 1e-6

# ── Canned corpus ───────────────────────────────────────────────────────────
# source -> (styled_version_0, styled_version_1). Valid Russian on both sides so
# perplexity / BERTScore / LaBSE stay finite. author='News' (not a TST target
# below) so every source survives the `author != target_style` filter on both
# text-target-style passes.
SOURCE_AUTHOR = "News"
TST_STYLES = ["Tolstoy", "Dostoevsky"]   # text-target-style path
EMB_DESC = "Tolstoy"                       # target-style-emb path

CANNED = {
    "Сегодня утром в городе прошёл сильный дождь.":
        ("Поутру над городом разразился проливной дождь.",
         "Утром на город обрушился сильный ливень."),
    "Учёные представили новое исследование о климате.":
        ("Мужи науки явили миру новый труд о климате.",
         "Исследователи обнародовали свежую работу о климате."),
    "Мальчик медленно шёл по пустой улице.":
        ("Отрок неспешно брёл по опустевшей улице.",
         "Юноша тихо шагал вдоль безлюдной улицы."),
    "Она открыла окно и посмотрела на сад.":
        ("Она отворила окно и взглянула на сад.",
         "Распахнув окно, она устремила взор в сад."),
    "В деревне начали готовиться к долгой зиме.":
        ("В селе принялись готовиться к долгой зиме.",
         "Деревня стала собираться к суровой зиме."),
    "Старый дом стоял на краю леса.":
        ("Ветхий дом стоял на опушке леса.",
         "Старая изба ютилась у самого леса."),
}
SOURCES = list(CANNED.keys())

# Columns frozen in the golden (final scores + key per-metric signals).
FREEZE_COLS = [
    "style_score", "meaning_score", "naturality_score", "score",
    "meaning_nolen", "nat_v2", "bi_cl", "gender_score", "entity_score",
    "score_v2",
]

CASES = [
    # (case_name, path, n_versions)
    ("text_style_1d", "text", 1),
    ("text_style_2d", "text", 2),
    ("emb_1d",        "emb",  1),
    ("emb_2d",        "emb",  2),
]


# ── Harness builders ────────────────────────────────────────────────────────
def _make_tst_func(n_versions):
    """Canned, deterministic. Handles both call signatures:
    text-target-style path calls (texts, target_style) positionally;
    target-style-emb path calls (texts=..., target_style_embeddings=...).
    Styling is canned, so both style arguments are ignored."""
    def f(texts, target_style=None, target_style_embeddings=None):
        if n_versions == 1:
            return [CANNED[t][0] for t in texts]
        return [[CANNED[t][v] for v in range(n_versions)] for t in texts]
    return f


def _build_test_df(path):
    if path == "text":
        return pd.DataFrame({"text": SOURCES, "author": [SOURCE_AUTHOR] * len(SOURCES)})
    elif path == "emb":
        return pd.DataFrame({
            "text": SOURCES,
            "author": [SOURCE_AUTHOR] * len(SOURCES),
            "target_style_key": [EMB_DESC] * len(SOURCES),
            "target_style_desc": [EMB_DESC] * len(SOURCES),
        })
    raise ValueError(path)


def _sort_and_freeze(df):
    """Deterministic row order + frozen columns as a list of dicts."""
    style_col = "target_style" if "target_style" in df.columns else "target_style_desc"
    sort_cols = ["example_number", style_col, "version_number"]
    sub = df.sort_values(sort_cols).reset_index(drop=True)
    rows = []
    for _, r in sub.iterrows():
        rec = {
            "example_number": int(r["example_number"]),
            "style": str(r[style_col]),
            "version_number": int(r["version_number"]),
        }
        for c in FREEZE_COLS:
            rec[c] = float(r[c])
        rows.append(rec)
    return rows


def _best_selection(pm, score_col):
    pm.select_best_tst_version(score_col=score_col)
    best = pm.best_tst_results
    style_col = "target_style" if "target_style" in best.columns else "target_style_desc"
    return {
        f"{int(r['example_number'])}|{r[style_col]}": int(r["version_number"])
        for _, r in best.iterrows()
    }


def _run_case(path, n_versions, author_styles, scorers):
    align_scorer, gender_scorer, entity_scorer = scorers
    df = _build_test_df(path)
    target_styles = TST_STYLES if path == "text" else None
    # The text path scores against author_styles[target_style]; the emb path
    # selects its target by target_style_key from the same labelled source.
    pm = TstPerformanceMetrics(
        test_df=df,
        tst_func=_make_tst_func(n_versions),
        target_styles=target_styles,
        tst_model="golden",
        author_styles=author_styles if path == "text" else None,
        target_style_source=author_styles if path == "emb" else None,
        verbose=False,
    )
    pm.execute()                                  # v1: produce + compute_scores + select_best
    best_v1 = _best_selection(pm, "score")        # capture v1 selection
    pm.compute_quality_scores(align_scorer, gender_scorer, entity_scorer)
    pm.compute_composite_v2()                     # score_v2
    best_v2 = _best_selection(pm, "score_v2")     # capture v2 selection (overwrites best_tst_results)

    rows = _sort_and_freeze(pm.tst_results)
    for rec in rows:
        for c in FREEZE_COLS:
            assert np.isfinite(rec[c]), f"non-finite {c} in case row {rec}"
    return {"rows": rows, "best_v1": best_v1, "best_v2": best_v2}


def _build_scorers():
    from tst_utils.eval.metrics.alignment import BatchedAligner
    from tst_utils.eval.metrics.gender_consistency import GenderConsistencyScorer
    from tst_utils.eval.metrics.entity_consistency import EntityConsistencyScorer
    align_scorer = BatchedAligner(
        model="ai-forever/ruRoberta-large", layer=8, method="itermax",
        score_fn="rolling_max", window=9, device="cuda",
        fallback=BatchedAligner(
            model="cointegrated/rubert-tiny2", layer=3, method="itermax",
            score_fn="span", min_span=4, device="cuda"),
    )
    gender_scorer = GenderConsistencyScorer(device="cuda")
    entity_scorer = EntityConsistencyScorer(device="cuda", pre_clean=False)
    return align_scorer, gender_scorer, entity_scorer


def _load_author_styles():
    """The canonical author_styles at the pin, raw and labelled."""
    return load_labelled_author_styles()


def run_all_cases():
    """Build scorers once, run all four cases. Returns {case_name: result}."""
    author_styles = _load_author_styles()
    scorers = _build_scorers()
    out = {}
    for name, path, nver in CASES:
        out[name] = _run_case(path, nver, author_styles, scorers)
        gc.collect()
        torch.cuda.empty_cache()
    return out


# ── pytest ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def live():
    return run_all_cases()


@gpu_only
@pytest.mark.parametrize("case_name", [c[0] for c in CASES])
def test_golden_parity(case_name, golden, live):
    g = golden["cases"][case_name]
    l = live[case_name]

    assert len(l["rows"]) == len(g["rows"]), f"{case_name}: row count drift"
    for i, (gr, lr) in enumerate(zip(g["rows"], l["rows"])):
        assert lr["example_number"] == gr["example_number"], (case_name, i)
        assert lr["style"] == gr["style"], (case_name, i)
        assert lr["version_number"] == gr["version_number"], (case_name, i)
        for c in FREEZE_COLS:
            assert abs(lr[c] - gr[c]) < TOL, (
                f"{case_name} row {i} col {c}: live={lr[c]} golden={gr[c]}")

    assert l["best_v1"] == g["best_v1"], f"{case_name}: v1 selection drift"
    assert l["best_v2"] == g["best_v2"], f"{case_name}: v2 selection drift"


@pytest.mark.parametrize("pin", ["base_v1", "base_v2"])
def test_inline_target_column_follows_the_pin(monkeypatch, pin):
    """CPU: an inline target_style_emb column warns while the pin is base_v1
    and raises once it is not (task 5.2). The encoder load seam is a fake,
    so no model is fetched or loaded."""
    from tst_utils.eval import model_names
    from tst_utils.eval.metrics import style as style_module

    class _Fake:
        def encode(self, texts, show_progress_bar=False,
                   normalize_embeddings=False):
            return np.full((len(list(texts)), 8), 15.0, dtype=np.float32)

    monkeypatch.setattr(model_names, "STYLE_ENCODER_KEY", pin)
    monkeypatch.setattr(style_module, "load_style_encoder", lambda key: _Fake())
    df = pd.DataFrame({"text": SOURCES[:2], "author": [SOURCE_AUTHOR] * 2,
                       "target_style_emb": [np.ones(8, dtype=np.float32)] * 2,
                       "target_style_desc": [EMB_DESC] * 2})
    pm = TstPerformanceMetrics(test_df=df, tst_func=_make_tst_func(1),
                               target_styles=None, tst_model="golden",
                               author_styles=None)
    if pin == "base_v1":
        with pytest.warns(StyleProvenanceWarning, match="target_style_emb"):
            pm.produce_tst_results()
    else:
        with pytest.raises(StyleLabelError, match="target_style_emb"):
            pm.produce_tst_results()


# ── generation ──────────────────────────────────────────────────────────────
def generate_golden(out_dir=None, note=None):
    import time
    from tst_utils.eval import model_names
    cases = run_all_cases()
    path = os.path.join(out_dir, "performance_golden.json") if out_dir else GOLDEN
    task = "2A.5 performance.py refactor"
    if note:
        task = f"{task}; {note}"
    out = {
        "meta": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "evaluation_encoder": model_names.style_encoder_key(),
            "tst_func": "canned-deterministic",
            "frozen_cols": FREEZE_COLS,
            "scorers": ("ruRoberta-L8 rolling_max w9 + rubert-tiny2-L3 span4 fallback; "
                        "GenderConsistencyScorer; EntityConsistencyScorer(pre_clean=False)"),
            "note": "behaviour-preserving golden; freezes CURRENT behaviour, bugs included",
        },
        "cases": cases,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    n = sum(len(c["rows"]) for c in cases.values())
    print(f"Golden written -> {path} ({len(cases)} cases, {n} rows)")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="the performance golden")
    parser.add_argument("--generate", action="store_true",
                        help="(re)write the golden JSON")
    parser.add_argument("--out-dir", default=None,
                        help="write performance_golden.json here "
                             "(default: this folder)")
    parser.add_argument("--note", default=None,
                        help="appended to meta.task")
    args = parser.parse_args(argv)
    if not args.generate:
        print("Pass --generate to (re)write the golden JSON.")
        return 0
    generate_golden(out_dir=args.out_dir, note=args.note)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
