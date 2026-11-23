"""The text-unit disjointness gate: one definition of the norm-key.

A disjointness claim is scoped to the unit its split was cut on, and the units
that were NOT cut on have to be measured. This module is the measuring side of
that rule for the text unit: it reduces a text to a normalized md5 key, and it
builds the key set of a list of training sources.

WHY IT IS SHARED. The gate lived in the style-v2 workstream
(`17 phase style-v2/_retrain_data/step4_build_dev.py:train_norm_keys`) and read
fixed paths, so a second consumer with its own training set could not use it and
had to restate it. The 2B GPT mix in `docs/tasks/2026-11-23-2b-train-gpt/` is
that consumer. One definition, one set of counts.

The rule that makes the gate necessary at all:
`docs/issues/disjointness-holds-only-at-the-cut-unit.md`.
"""
import hashlib
import re

import pandas as pd


def norm_key(text):
    """Normalization used for md5 dedup keys (matches pool_v2_chunk).

    Moved unchanged from `17 phase style-v2/_retrain_data/common.py`, which now
    re-exports it; this is the only definition.
    """
    return hashlib.md5(re.sub(r'\W+', '', text.lower()).encode()).hexdigest()


def norm_keys_of(sources):
    """Build the norm-key set of a list of training sources, with per-source counts.

    `sources` is a list of tuples, either `(path, columns)` or
    `(path, columns, row_filter)`:
      - `columns` is the list of text columns to key;
      - `row_filter` is `(filter_columns, fn)`. The file is read with
        `columns + filter_columns`, and `fn(df)` returns a boolean mask of the
        rows to keep. Use it when a source is a full table of which only some
        rows train.
    Every non-null value of every listed text column is keyed.

    Returns `(keys, counts)`: `keys` is the union set, and `counts` is one dict
    per source in the order given — `path`, `columns`, `rows_read`,
    `rows_kept`, `keys` (distinct keys this source contributed) and
    `union_after` (size of the union once this source is added). No logging:
    the caller decides what to print.
    """
    keys = set()
    counts = []
    for source in sources:
        path, columns = source[0], list(source[1])
        row_filter = source[2] if len(source) > 2 else None

        read_columns = list(columns)
        if row_filter is not None:
            filter_columns, fn = row_filter
            read_columns += [c for c in filter_columns if c not in read_columns]
        df = pd.read_parquet(path, columns=read_columns)

        rows_read = len(df)
        if row_filter is not None:
            df = df[fn(df).fillna(False).astype(bool)]

        source_keys = set()
        for col in columns:
            for t in df[col].dropna():
                source_keys.add(norm_key(str(t)))
        keys |= source_keys
        counts.append({
            "path": path,
            "columns": list(columns),
            "rows_read": rows_read,
            "rows_kept": len(df),
            "keys": len(source_keys),
            "union_after": len(keys),
        })
    return keys, counts
