import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from collections import Counter


def chrf_scores(texts, styled_texts):
    """Compute chrF scores for pairs of texts using sacrebleu."""
    scores = []
    for src, tgt in zip(texts, styled_texts):
        score = sacrebleu.sentence_chrf(tgt, [src])
        scores.append(score.score / 100.0)
    return np.array(scores)


def bleu_scores(texts, styled_texts):
    """Compute sentence-level BLEU scores for pairs of texts using sacrebleu."""
    scores = []
    for src, tgt in zip(texts, styled_texts):
        score = sacrebleu.sentence_bleu(tgt, [src], smooth_method='exp')
        scores.append(score.score / 100.0)
    return np.array(scores)


def rouge_l_scores(texts, styled_texts):
    """Compute ROUGE-L F1 scores for pairs of texts."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []
    for src, tgt in zip(texts, styled_texts):
        score = scorer.score(src, tgt)
        scores.append(score['rougeL'].fmeasure)
    return np.array(scores)


def _word_ngrams(text, n):
    """Extract word n-grams from text."""
    words = text.split()
    if len(words) < n:
        return Counter()
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def ngram_jaccard_overlap(texts, styled_texts, n=2):
    """Compute word n-gram Jaccard overlap between text pairs."""
    scores = []
    for src, tgt in zip(texts, styled_texts):
        src_ngrams = _word_ngrams(src, n)
        tgt_ngrams = _word_ngrams(tgt, n)
        if not src_ngrams and not tgt_ngrams:
            scores.append(0.0)
            continue
        intersection = sum((src_ngrams & tgt_ngrams).values())
        union = sum((src_ngrams | tgt_ngrams).values())
        scores.append(intersection / union if union > 0 else 0.0)
    return np.array(scores)


def compute_all_copying_metrics(texts, styled_texts):
    """Compute all copying metrics and return as a dict of numpy arrays."""
    return {
        'chrf': chrf_scores(texts, styled_texts),
        'bleu': bleu_scores(texts, styled_texts),
        'rouge_l': rouge_l_scores(texts, styled_texts),
        'ngram_jaccard_2': ngram_jaccard_overlap(texts, styled_texts, n=2),
        'ngram_jaccard_3': ngram_jaccard_overlap(texts, styled_texts, n=3),
        'ngram_jaccard_4': ngram_jaccard_overlap(texts, styled_texts, n=4),
    }
