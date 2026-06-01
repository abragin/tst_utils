from sentence_transformers import SentenceTransformer
from tst_utils.eval.model_names import STYLE_MODEL
import numpy as np


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
    model = SentenceTransformer(STYLE_MODEL)
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
