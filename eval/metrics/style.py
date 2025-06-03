from sentence_transformers import SentenceTransformer
from tst_utils.eval.model_names import STYLE_MODEL
import numpy as np


def calc_style_embeddings(texts):
    model = SentenceTransformer(STYLE_MODEL)
    return [e for e in model.encode(list(texts), show_progress_bar=True)]

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

def add_away_towards(df, author_styles):
    away_val = []
    towards_val = []
    for _, row in df.iterrows():
        source = row.text_emb
        current = row.styled_text_emb
        target = author_styles[row.target_style]
        away_val.append(away(source, current, target))
        towards_val.append(towards(source, current, target))
    df['away'] = away_val
    df['towards'] = towards_val
