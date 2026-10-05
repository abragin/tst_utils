import numpy as np
import torch
from tst_utils.datasets.utils import a2tag, produce_segments

# One sentence, encoded with the tokenizer's default special tokens, tells how
# the tokenizer ends a sequence. `add_special_tokens=False` would hide the EOS
# of a T5-family tokenizer, so the probe never uses it.
TOKENIZER_PROBE_SENTENCE = 'Это тестовое предложение.'


def check_tokenizer_matches_model_type(tokenizer, model_type):
    """Raise ValueError if `tokenizer` does not fit the `model_type` code path.

    The 'T5' path slices the last id of every sentence as if it were EOS, so
    the tokenizer must end a sentence with EOS. The 'GPT' path keeps every id
    and joins sentences with `encode(' ')[0]`, so the tokenizer must not.
    """
    name = getattr(tokenizer, 'name_or_path', None) or type(tokenizer).__name__
    eos_id = tokenizer.eos_token_id
    ends_with_eos = tokenizer.encode(TOKENIZER_PROBE_SENTENCE)[-1] == eos_id
    if model_type == 'T5' and not ends_with_eos:
        raise ValueError(
            f"ChapterDataset: tokenizer {name!r} does not end an encoded "
            "sentence with its EOS token, but model_type='T5' slices the last "
            "id of every sentence as if it were EOS. With this tokenizer that "
            "removes the last real token (usually the final period), and the "
            "sentences of a chunk are joined with no separator, because only "
            "the 'GPT' path adds a space token. Pass a T5-family tokenizer, "
            "or pass allow_tokenizer_mismatch=True to reproduce an existing "
            "build on purpose."
        )
    if model_type == 'GPT' and ends_with_eos:
        raise ValueError(
            f"ChapterDataset: tokenizer {name!r} ends every encoded sentence "
            "with its EOS token, but model_type='GPT' keeps every id and "
            "joins the sentences with `tokenizer.encode(' ')[0]`. With this "
            "tokenizer an EOS id stands inside the sequence after every "
            "sentence, and the joining 'space' is an EOS id. Pass a GPT-style "
            "tokenizer, or pass allow_tokenizer_mismatch=True to reproduce an "
            "existing build on purpose."
        )


class ChapterDataset(torch.utils.data.Dataset):
    def __init__(
        self, chapter_df, tokenizer, source_cols,
        length_sampler,
        model_type,
        max_tok_len = None, # Disired max token length of source and target texts (not guaranteed)
        target_col = 'text_ru',
        style_vector = None,
        *,
        max_side_length = None, # per-side token limit (source and target each truncated to it; for GPT they are concatenated so the training sequence is ~2x this)
        max_length = None, # renamed -> max_side_length; sentinel for the old name
        allow_tokenizer_mismatch = False, # skip the tokenizer/model_type check; only to reproduce an old build on purpose
    ):
        if max_length is not None:
            raise ValueError(
                "ChapterDataset: `max_length` was renamed to `max_side_length` "
                "(per-side token limit; for GPT, source and target are "
                "concatenated so the training sequence is ~2x this). "
                "Pass max_side_length=... instead."
            )
        if max_side_length is None:
            raise ValueError(
                "ChapterDataset: `max_side_length` is required "
                "(per-side token limit)."
            )
        self.target_col = target_col
        self.texts_target = chapter_df[self.target_col].tolist()
        present_source_cols = [c for c in source_cols if (chapter_df[c] != '').any()]
        self.texts_source = {
            col_name: chapter_df[col_name].tolist()
            for col_name in present_source_cols
        }
        self.author_tag = a2tag(chapter_df.author.iloc[0]) if 'author' in chapter_df else None
        self.tokenizer = tokenizer
        self.length_sampler = length_sampler
        if model_type in ['GPT', 'T5']:
            self.model_type = model_type
        else:
            raise Exception("Unsupported model type: ", model_type)
        self.max_side_length = max_side_length
        if tokenizer.eos_token_id is None:
            # `build_input` appends `eos_token_id` on both paths, so a missing
            # one would put `None` into the ids. The keyword does not excuse it.
            raise ValueError(
                "ChapterDataset: the tokenizer has no `eos_token_id`, and "
                "`build_input` appends it to every sequence."
            )
        if not allow_tokenizer_mismatch:
            check_tokenizer_matches_model_type(tokenizer, model_type)

        if model_type == 'T5':
            ids_up_bound = -1
        else:
            ids_up_bound  = None
            self.space_encoded = tokenizer.encode(' ')[0]
        self.target_input_ids = [
            iids[:ids_up_bound] # Skip EOS token in case of T5
            for iids in tokenizer(
                self.texts_target, truncation=True, max_length=max_side_length
            )['input_ids']
        ]
        self.source_input_ids = {
            c: [
                    iids[:ids_up_bound] # Skip EOS token in case of T5
                    for iids in tokenizer(
                        self.texts_source[c], truncation=True, max_length=max_side_length
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
            max(1,round(self.tot_tokens / self.length_sampler.expected_mean())),
            chapter_df.shape[0]
        )
        self.sample()
        self.eos_token_id =  tokenizer.eos_token_id

    def build_input(self, source_ids, target_ids):
        if self.model_type == 'T5':
            if self.author_token_ids:
                source_ids = self.author_token_ids + source_ids
            if len(source_ids) > (self.max_side_length-1):
                source_ids = source_ids[:self.max_side_length-1]
            if len(target_ids) > (self.max_side_length-1):
                target_ids = target_ids[:self.max_side_length-1]

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
            if len(source_ids) > (self.max_side_length-2):
                source_ids = source_ids[:self.max_side_length-1]
            if len(target_ids) > (self.max_side_length-2):
                target_ids = target_ids[:self.max_side_length-1]
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
            self.length_sampler
        )
        if self.multiple_sources:
            self.selected_options = [
                np.random.choice(so) for so in self.source_options
            ]
        else:
            self.selected_options = [so[0] for so in self.source_options]