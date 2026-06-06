import numpy as np
import sacrebleu


def chrf_scores(texts, styled_texts):
    """Compute chrF scores for pairs of texts using sacrebleu."""
    scores = []
    for src, tgt in zip(texts, styled_texts):
        score = sacrebleu.sentence_chrf(tgt, [src])
        scores.append(score.score / 100.0)
    return np.array(scores)
