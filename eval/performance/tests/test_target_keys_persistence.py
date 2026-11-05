"""Integration test for `target_keys` survival through the results pipeline.

Covers the gap between the unit tests (producers / parquet round-trip) and the
GPU end-to-end run: that `target_keys`, once present on `test_df`, survives the
`cols_to_copy` whitelist in `produce_tst_results_with_target_style_emb` and the
`select_best_tst_version` reduction — the exact `core.py` edge a single-whitelist
patch would silently drop. GPU-free: `tst_func` is mocked, no models load.
"""
import numpy as np
import pandas as pd

from tst_utils.eval.data.style_store import LabelledStyleVectors
from tst_utils.eval.performance.core import TstPerformanceMetrics

# Task 5.2: targets are selected by `target_style_key` from a labelled source.
# The sampled identities (`target_keys`) and the scalar lookup key
# (`target_style_key`) are separate columns and stay separate.
TARGETS = LabelledStyleVectors.declare(
    [np.full(8, 14.0, dtype=np.float32), np.arange(8, dtype=np.float32) + 10],
    "base_v1", "target-keys probe", keys=["t_bible", "t_mix"])


def _make_pm(test_df, tst_output, seen=None):
    def tst_func(texts, target_style_embeddings):
        assert len(texts) == len(test_df)
        if seen is not None:
            seen.append(target_style_embeddings.row_keys)
        return tst_output
    return TstPerformanceMetrics(
        test_df=test_df, tst_func=tst_func, target_styles=None,
        tst_model="mock-model", author_styles=None,
        evaluation_encoder="base_v1", target_style_source=TARGETS,
    )


def _emb_test_df():
    # Two source rows with distinct target_keys (n_picks 1 and 2).
    return pd.DataFrame({
        "text": ["s0", "s1"],
        "author": ["a0", "a1"],
        "target_style_key": ["t_bible", "t_mix"],
        "target_style_desc": ["other_domain", "2_domains_weighted_avg"],
        "target_keys": [["Bible"], ["Bible", "news"]],
    })


def test_target_keys_survives_cols_to_copy_and_selection(fake_encoder):
    test_df = _emb_test_df()
    # 2 generated versions per example -> exercises the 2D expand path + selection.
    tst_output = [["v00", "v01"], ["v10", "v11"]]
    pm = _make_pm(test_df, tst_output)

    pm.produce_tst_results_with_target_style_emb()

    # (1) survives the cols_to_copy whitelist into the long results frame
    assert "target_keys" in pm.tst_results.columns
    assert pm.tst_results["target_keys"].notna().all()
    # every version-row of an example carries that example's keys
    for ex_num, grp in pm.tst_results.groupby("example_number"):
        keys = [list(v) for v in grp["target_keys"]]
        assert all(k == keys[0] for k in keys)
    by_example = {
        ex: [list(v) for v in grp["target_keys"]][0]
        for ex, grp in pm.tst_results.groupby("example_number")
    }
    assert by_example == {0: ["Bible"], 1: ["Bible", "news"]}

    # (2) survives select_best_tst_version's groupby-idxmax reduction. Score the
    # rows so a specific version wins per example; the winner must keep its keys.
    score_map = {("v00"): 0.1, ("v01"): 0.9, ("v10"): 0.8, ("v11"): 0.2}
    pm.tst_results["score"] = pm.tst_results["styled_text"].map(score_map)
    pm.select_best_tst_version(score_col="score")

    best = pm.best_tst_results
    assert "target_keys" in best.columns
    assert best["target_keys"].notna().all()
    assert len(best) == 2  # one winner per example
    winners = dict(zip(best["example_number"], best["styled_text"]))
    assert winners == {0: "v01", 1: "v10"}
    keys_by_example = {
        ex: list(k) for ex, k in zip(best["example_number"], best["target_keys"])
    }
    assert keys_by_example == {0: ["Bible"], 1: ["Bible", "news"]}

    # (3) the lookup key is a separate column and survives both steps too
    lookup = dict(zip(best["example_number"], best["target_style_key"]))
    assert lookup == {0: "t_bible", 1: "t_mix"}
    assert set(pm.tst_results["target_style_key"]) == {"t_bible", "t_mix"}


def test_no_target_keys_column_does_not_break_or_fabricate(fake_encoder):
    # The core.py add is conditional: an eval-only test_df without target_keys
    # must neither raise nor invent the column.
    test_df = _emb_test_df().drop(columns=["target_keys"])
    pm = _make_pm(test_df, [["v00", "v01"], ["v10", "v11"]])
    pm.produce_tst_results_with_target_style_emb()
    assert "target_keys" not in pm.tst_results.columns
    assert "target_style_key" in pm.tst_results.columns


def test_the_lookup_key_reaches_tst_func_in_row_order(fake_encoder):
    seen = []
    pm = _make_pm(_emb_test_df(), ["x0", "x1"], seen=seen)
    pm.produce_tst_results()
    assert seen == [("t_bible", "t_mix")]
