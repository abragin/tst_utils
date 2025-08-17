import numpy as np
import torch
import pandas as pd
from itertools import cycle
from tst_utils.datasets.books_dataset import BooksIterableDataset


class NewsDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, news_files, tokenizer, model_type,
        source_cols=['text_ru_pph'],
        target_col = 'text_ru',
        min_tok_len=50, avg_tok_len=150, max_length = 512,
        max_tok_len=None,
        style_emb = None,
        debug = False
    ):
        self.news_files = news_files
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.source_cols = source_cols
        self.target_col = target_col
        self.min_tok_len = min_tok_len
        self.news_file_ind = 0
        self.avg_tok_len = avg_tok_len
        self.max_tok_len = max_tok_len
        self.max_length = max_length
        self.style_emb = style_emb
        self.debug = debug

    def load_next_news_file(self, file):
        compr = 'gzip' if file.endswith('.gz') else None
        news_df = pd.read_csv(file, compression=compr)
        news_df['author'] = 'News'
        style_dict = {'News': self.style_emb} if self.style_emb is not None else None
        self.current_ds = BooksIterableDataset(
            news_df,
            tokenizer = self.tokenizer,
            source_cols = self.source_cols,
            model_type = self.model_type,
            target_col = self.target_col,
            min_tok_len = self.min_tok_len,
            avg_tok_len = self.avg_tok_len,
            max_tok_len = self.max_tok_len,
            max_length = self.max_length,
            style_dict = style_dict,
            debug = self.debug
        )
        self.current_ds.max_samples = len(self.current_ds.books_dataset)

    def __iter__(self):
        for f in cycle(self.news_files):
            self.load_next_news_file(f)
            news_iter = iter(self.current_ds)
            yield from news_iter   