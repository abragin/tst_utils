"""chrF copying diagnostic (kept separate from the composite score)."""

from tst_utils.eval.metrics.copying import chrf_scores


def add_chrf(df):
    """Add a ``chrf`` column to ``df`` in place (source ``text`` vs ``styled_text``)
    and return it. Does NOT modify the composite score. chrF is the only retained
    copying metric (task 1.3: BLEU/ROUGE-L/n-gram Jaccard were redundant or broken).
    """
    df["chrf"] = chrf_scores(df.text, df.styled_text)
    return df
