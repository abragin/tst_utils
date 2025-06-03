import numpy as np
import torch
from tst_utils.datasets.chapter_dataset import ChapterDataset

class BooksDataset(torch.utils.data.Dataset):
    def __init__(
        self, books_df, tokenizer, source_cols,
        model_type, target_col = 'text_ru',
        min_tok_len=50, avg_tok_len=150, max_tok_len=500, max_length = 512,
    ):
        chapters = books_df[['title','author', 'chapter_pos']].drop_duplicates()
        self.chapter_datasets = []
        for _, chapter_row in chapters.iterrows():
            chapter_df = books_df[(
                (books_df.title == chapter_row.title) &
                (books_df.chapter_pos == chapter_row.chapter_pos) &
                (books_df.author == chapter_row.author)
            )]
            self.chapter_datasets.append(ChapterDataset(
                chapter_df,
                tokenizer = tokenizer,
                source_cols = source_cols,
                model_type = model_type,
                min_tok_len = min_tok_len,
                avg_tok_len = avg_tok_len,
                max_tok_len = max_tok_len,
                max_length = max_length
            ))
        self.total_length = sum(len(dataset) for dataset in self.chapter_datasets)
        self.positions = []
        for i, dataset in enumerate(self.chapter_datasets):
            for j in range(len(dataset)):
                self.positions.append((i, j))

    def sample(self):
        # Chapter datasets call `sample` on init, so this method
        # shouldn't be called on init
        for cds in self.chapter_datasets:
            cds.sample()

    def debug_info(self, ind):
        # ind = self.index[self.cur % self.total_length]
        dataset_index, chapter_index = self.positions[ind]
        chapter_dataset = self.chapter_datasets[dataset_index]
        rng_start, rng_end = chapter_dataset.segment_ranges[chapter_index]
        selected_options = chapter_dataset.selected_options[rng_start:rng_end]
        return {
            'dataset_index': dataset_index,
            'chapter_index': chapter_index,
            'range': (rng_start, rng_end),
            'selected_options': selected_options
        }

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        dataset_index, chapter_index = self.positions[index]
        return self.chapter_datasets[dataset_index][chapter_index]

class BooksIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, books_df, tokenizer, source_cols,
        model_type, target_col = 'text_ru',
        min_tok_len=50, avg_tok_len=150, max_tok_len=500, max_length = 512,
        max_samples=None, debug=False
    ):
        self.books_dataset = BooksDataset(
            books_df,
            tokenizer = tokenizer,
            source_cols = source_cols,
            model_type = model_type,
            target_col = target_col,
            min_tok_len = min_tok_len,
            avg_tok_len = avg_tok_len,
            max_tok_len=max_tok_len,
            max_length = max_length,
        )
        self.index = list(range(self.books_dataset.total_length))
        np.random.shuffle(self.index)
        self.cur = 0
        self.max_samples = max_samples
        self.debug = debug

    def sample(self):
        # Chapter datasets call `sample` on init, so this method
        # shouldn't be called if self.cur == 0
        np.random.shuffle(self.index)
        self.books_dataset.sample()

    def ind(self):
        return self.index[self.cur % self.books_dataset.total_length]

    def debug_info(self):
        return self.books_dataset.debug_info(self.ind())

    def __iter__(self):
        b_ds = self.books_dataset
        while True:
            if self.max_samples is not None and self.cur >= self.max_samples:
                break
            if (self.cur % b_ds.total_length) == 0 and self.cur > 0:
                self.sample()
            res = b_ds[self.ind()]
            if self.debug:
                yield (res, self.debug_info())
            else:
                yield res
            self.cur += 1
