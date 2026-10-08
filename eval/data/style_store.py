"""Labels and guarded IO for stored style embeddings.

A stored style embedding carries its label: the encoder that produced it
and its normalization. Readers check the label against the pin before
they return a vector.
"""

import hashlib
import warnings

import numpy as np


class StyleProvenanceWarning(FutureWarning):
    """A stored style vector is used on provenance assumptions.

    This is a `FutureWarning` and not a `DeprecationWarning`, because
    Python hides a `DeprecationWarning` that fires outside `__main__`
    (PEP 565), and these warnings fire inside `tst_utils`, so a script
    user would not see them. Repeats are removed by the standard warnings
    registry, with no state of our own: the default action shows each
    distinct message once per call site, so each message names its column
    or its npz file and its entry point.
    """


def style_text_key(text):
    """Return the join key of `text`: hex sha256 of its UTF-8 bytes.

    The input is the exact string the encoder receives.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def measure_normalization(embeddings):
    """Measure the normalization of per-row `embeddings`.

    Returns a dict with `normalization` (`"l2"` if every norm is within
    1e-3 of 1, `"none"` otherwise) plus `norm_min` and `norm_max` as
    Python floats. The norms are computed in float64 whatever the input
    dtype.

    Raises:
        ValueError: on an empty input; if some norms are within 1e-3 of 1
            and others are not (a mixed-scale array); or if
            `norm_max / norm_min` exceeds 2.
    """
    norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float64), axis=1)
    if norms.size == 0:
        raise ValueError("measure_normalization needs at least one row")
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    within = np.abs(norms - 1.0) <= 1e-3
    if np.any(within) and not np.all(within):
        raise ValueError(
            "mixed-scale embeddings: some norms are within 1e-3 of 1 "
            f"(min {norm_min}, max {norm_max})"
        )
    if norm_min == 0.0:
        if norm_max > 0.0:
            raise ValueError(
                "mixed-scale embeddings: zero and non-zero norms "
                f"(max {norm_max})"
            )
    elif norm_max / norm_min > 2:
        raise ValueError(
            f"mixed-scale embeddings: norm_max / norm_min exceeds 2 "
            f"(min {norm_min}, max {norm_max})"
        )
    return {
        "normalization": "l2" if bool(np.all(within)) else "none",
        "norm_min": norm_min,
        "norm_max": norm_max,
    }


def warn_provenance(message):
    """Emit a `StyleProvenanceWarning` with stacklevel 2."""
    warnings.warn(message, StyleProvenanceWarning, stacklevel=2)
