"""Shared constants for the performance-evaluation package."""

# Columns required by compute_composite_v2 before it can build score_v2.
_SCORE_COLS = ['style_score', 'bert_score', 'labse_score',
               'text_perplexity', 'styled_text_perplexity']
_QUALITY_COLS = ['bi_cl', 'gender_score', 'entity_score']


TARGET_STYLES = {
    'DT': ['Dostoevsky', 'Tolstoy'],
    'BCDT': ['Bible', 'Chekhov', 'Dostoevsky', 'Tolstoy'],
    'COMPLETE': ['Bible', 'Chekhov', 'Dostoevsky', 'Tolstoy', 'News'],
}
