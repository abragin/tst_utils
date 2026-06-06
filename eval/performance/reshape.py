"""Reshape ``tst_func`` output into a long results DataFrame."""

import pandas as pd


def expand_tst_output(tst_output, df, cols_to_copy, extra_fields) -> pd.DataFrame:
    """Reshape ``tst_func`` output into a long DataFrame (one row per
    (example, version)). Handles both shapes returned by ``tst_func``:

      * 2D ``[[ver0, ver1, ...], ...]`` — one record per version, columns from
        ``cols_to_copy`` plus ``extra_fields`` plus styled_text/version.
      * 1D ``[text, ...]`` — single version (version_number=0).

    ``extra_fields`` is merged into every record (e.g. ``tst_model`` and, for the
    text-target-style path, ``target_style``).
    """
    if not tst_output:
        raise ValueError("tst_func returned an empty result.")

    if isinstance(tst_output[0], list):  # 2D: multiple versions per example
        num_versions = len(tst_output[0])
        if not all(len(x) == num_versions for x in tst_output):
            raise ValueError("Inconsistent number of generated versions per example.")
        flat_records = []
        for ex_idx, versions in enumerate(tst_output):
            for v_idx, text in enumerate(versions):
                flat_records.append(
                    df.loc[ex_idx, cols_to_copy].to_dict()
                    | extra_fields
                    | {"styled_text": text, "version_number": v_idx}
                )
        return pd.DataFrame(flat_records)

    # 1D: single output per example
    if len(tst_output) != len(df):
        raise ValueError(
            f"Length mismatch: tst_func returned {len(tst_output)} texts "
            f"for {len(df)} examples."
        )
    out = df.copy()
    out["styled_text"] = tst_output
    out["version_number"] = 0
    for k, v in extra_fields.items():
        out[k] = v
    return out
