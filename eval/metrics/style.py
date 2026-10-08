import hashlib
import os

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from tst_utils.eval.model_names import STYLE_ENCODER_KEY
from tst_utils.eval.style_encoder_registry import (
    get_encoder,
    get_excluded_filenames,
)
import numpy as np


class StyleEncoderIntegrityError(RuntimeError):
    """A style encoder snapshot differs from its registry entry."""


_snapshot_hash_cache = {}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_snapshot_hashes(directory, expected_file_sha256, excluded_filenames):
    """Check every file of `directory` against `expected_file_sha256`.

    Args:
        directory: the snapshot directory to check.
        expected_file_sha256: map from relative path (POSIX) to sha256,
            from the encoder registry.
        excluded_filenames: basenames that are never hashed (README.md,
            .gitattributes).

    Returns:
        dict: the computed map from relative path to sha256.

    Raises:
        StyleEncoderIntegrityError: naming the file, if a hash differs,
            a registered file is missing, or a file is present that is
            neither registered nor excluded.

    The computed hashes are cached once per process per realpath of the
    directory (0.36 s for the 713 MB weights on tallin, measured 2026-10-08), and compared
    against the expected dict on every call, so a changed registry
    value still raises after an earlier good load.
    """
    real_directory = os.path.realpath(directory)
    computed = _snapshot_hash_cache.get(real_directory)
    if computed is None:
        computed = {}
        for root, _, filenames in os.walk(real_directory, followlinks=True):
            for name in filenames:
                path = os.path.join(root, name)
                relative = os.path.relpath(path, real_directory)
                computed[relative] = _sha256_file(os.path.realpath(path))
        _snapshot_hash_cache[real_directory] = computed
    for relative, expected in expected_file_sha256.items():
        if relative not in computed:
            raise StyleEncoderIntegrityError(
                f"registered file missing from snapshot: {relative}"
            )
        if computed[relative] != expected:
            raise StyleEncoderIntegrityError(
                f"hash mismatch for snapshot file: {relative}"
            )
    for relative in computed:
        if (
            relative not in expected_file_sha256
            and os.path.basename(relative) not in excluded_filenames
        ):
            raise StyleEncoderIntegrityError(
                f"unregistered file in snapshot: {relative}"
            )
    return dict(computed)


def load_style_encoder(key):
    """Load the style encoder registered under `key`, checked on load.

    Resolves the Hub snapshot first, checks every file of that directory
    against the registry entry, and loads the model from that path, so
    the bytes it checks are the bytes it loads.

    Raises:
        NotImplementedError: for a `local_only` entry, naming the machine
            that holds the bytes.
        StyleEncoderIntegrityError: if the snapshot differs from the entry.
    """
    entry = get_encoder(key)
    if "local_only" in entry:
        raise NotImplementedError(
            f"load not implemented for style encoder {key!r}; "
            f"the bytes live on {entry['local_only']}"
        )
    snapshot_path = snapshot_download(
        entry["hub_id"], revision=entry["revision"]
    )
    check_snapshot_hashes(
        snapshot_path, entry["file_sha256"], get_excluded_filenames()
    )
    return SentenceTransformer(snapshot_path)


def calc_style_embeddings(texts, *, normalize):
    """Encode `texts` with the project style encoder.

    Args:
        texts: iterable of strings to encode.
        normalize: REQUIRED keyword. If True, output vectors are L2-normalized
            to unit norm (delegated to SentenceTransformer's
            ``normalize_embeddings``). If False, raw encoder outputs are
            returned (typically norm ~15 for ``abragin/ruBert-style-base``).

    Returns:
        list[np.ndarray]: one 1D embedding per input text. List-of-arrays is
        preserved (rather than a 2D ndarray) so that the result can be assigned
        directly to a pandas Series column.

    Notes:
        - Folder-14 TinyStyler expects unit-norm style inputs; pre-folder-14
          checkpoints expect unnormalized (~15). Pick `normalize` accordingly.
        - The eval-side similarity metrics (`sim_measure`, `away`, `towards`)
          are scale-invariant (angular), so `normalize=False` is correct for
          evaluation pipelines that consume `author_styles.npz` (unnormalized).
    """
    model = load_style_encoder(STYLE_ENCODER_KEY)
    return [
        e for e in model.encode(
            list(texts),
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
    ]

def sim_measure(u,v):
    ac_inp = np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))
    sim = 1 - np.arccos(np.clip(ac_inp, -1, 1))/np.pi
    return (sim + 1)/2

def sim_c(u, v):
    return (1 - sim_measure(u,v))

def away(source, current, target):
    """
    Measures how much the current text has moved away from the source
    relative to the distance between source and target styles.
    """
    s_c_ts = sim_c(target, source)
    return min(sim_c(current, source), s_c_ts)/s_c_ts

def towards(source, current, target):
    return max(
        sim_measure(current, target) - sim_measure(source, target), 0
    )/sim_c(target, source)

def add_away_towards(df, author_styles=None):
    away_val = []
    towards_val = []
    for _, row in df.iterrows():
        source = row.text_style_emb
        current = row.styled_text_style_emb
        target = author_styles[row.target_style] if author_styles else row.target_style_emb
        away_val.append(away(source, current, target))
        towards_val.append(towards(source, current, target))
    df['away'] = away_val
    df['towards'] = towards_val
