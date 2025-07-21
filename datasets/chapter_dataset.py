import numpy as np
import torch
from tst_utils.datasets.utils import a2tag, produce_segments


class ChapterDataset(torch.utils.data.Dataset):
    def __init__(
        self, chapter_df, tokenizer, source_cols,
        min_tok_len, avg_tok_len, max_tok_len, max_length,
        model_type, target_col = 'text_ru',
        style_vector = None
    ):
        self.target_col = target_col
        self.texts_target = chapter_df[self.target_col].tolist()
        present_source_cols = [c for c in source_cols if (chapter_df[c] != '').any()]
        self.texts_source = {
            col_name: chapter_df[col_name].tolist()
            for col_name in present_source_cols
        }
        self.author_tag = a2tag(chapter_df.author.iloc[0]) if 'author' in chapter_df else None
        self.tokenizer = tokenizer
        self.min_tok_len = min_tok_len
        self.avg_tok_len = avg_tok_len
        self.max_tok_len = max_tok_len
        self.max_length = max_length
        if model_type in ['GPT', 'T5']:
            self.model_type = model_type
        else:
            raise Exception("Unsupported model type: ", model_type)
        if model_type == 'T5': 
            ids_up_bound = -1
        else:
            ids_up_bound  = None
            self.space_encoded = tokenizer.encode(' ')[0]
        self.target_input_ids = [
            iids[:ids_up_bound] # Skip EOS token in case of T5
            for iids in tokenizer(
                self.texts_target, truncation=True, max_length=max_length
            )['input_ids']
        ]
        self.source_input_ids = {
            c: [
                    iids[:ids_up_bound] # Skip EOS token in case of T5
                    for iids in tokenizer(
                        self.texts_source[c], truncation=True, max_length=max_length
                    )['input_ids']
            ]
            for c in present_source_cols
        }
        self.source_options = [
            [c for c in present_source_cols if self.texts_source[c][i]]
            for i in range(chapter_df.shape[0])
        ]

        self.selected_options = [so[0] for so in self.source_options]
        self.multiple_sources = any([len(so) > 1 for so in self.source_options])
        self.token_counts = [len(ii) for ii in self.target_input_ids]
        self.style_vector = style_vector
        self.author_token_ids = None

        if self.style_vector is None:
            if self.author_tag:
                if model_type == 'T5':
                    self.author_token_ids = tokenizer.encode(self.author_tag)[:-1]
                else:
                    self.author_token_ids = tokenizer.encode(' ' + self.author_tag + ' ')
            else:
                raise Exception("Neither style_vector nor author are provided")
        self.tot_tokens = sum(self.token_counts)
        self.n_chunks = min(
            max(1,round(self.tot_tokens / avg_tok_len)),
            chapter_df.shape[0]
        )
        self.sample()
        self.eos_token_id =  tokenizer.eos_token_id

    def build_input(self, source_ids, target_ids):
        if self.model_type == 'T5':
            if self.author_token_ids:
                source_ids = self.author_token_ids + source_ids
            if len(source_ids) > (self.max_length-1):
                source_ids = source_ids[:self.max_length-1]
            if len(target_ids) > (self.max_length-1):
                target_ids = target_ids[:self.max_length-1]

            sep_token_id = self.eos_token_id
            source_ids.append(sep_token_id)
            target_ids.append(sep_token_id)
            attention_mask = [1] * len(source_ids)
    
            return {
                'input_ids': source_ids,
                'attention_mask': attention_mask,
                'labels': target_ids,
                'style': self.style_vector if self.style_vector is not None else None
            }
        elif self.model_type == 'GPT':
            if len(source_ids) > (self.max_length-2):
                source_ids = source_ids[:self.max_length]
            if len(target_ids) > (self.max_length-2):
                target_ids = target_ids[:self.max_length-1]
            sep_token_id = self.eos_token_id
            input_ids = (
                source_ids
                + (self.author_token_ids if self.author_token_ids else [sep_token_id])
                + target_ids
                + [sep_token_id]
            )
            attention_mask = [1] * len(input_ids)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'style': self.style_vector if self.style_vector is not None else None
            }
        else:
            raise Exception("Unsupported model type: ", self.model_type)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        rng_start, rng_end = self.segment_ranges[idx]
        source_ids = []
        target_ids = []
        for i in range(rng_start, rng_end):
            selected_option = self.selected_options[i]
            source_ids += self.source_input_ids[selected_option][i]
            target_ids += self.target_input_ids[i]
            if (self.model_type == 'GPT') and (i < rng_end - 1):
                source_ids.append(self.space_encoded)
                target_ids.append(self.space_encoded)
        return self.build_input(source_ids, target_ids)

    def sample(self):
        self.segment_ranges = produce_segments(
            self.token_counts, self.n_chunks, 
            self.min_tok_len, self.max_tok_len
        )
        if self.multiple_sources:
            self.selected_options = [
                np.random.choice(so) for so in self.source_options
            ]
        else:
            self.selected_options = [so[0] for so in self.source_options]
