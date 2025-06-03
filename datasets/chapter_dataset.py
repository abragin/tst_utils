import numpy as np
import torch
from itertools import accumulate
from tst_utils.datasets.utils import a2tag


def get_scores_and_indx(cnk_pos_norm, sent_pos_norm):
    n_rows = len(cnk_pos_norm)
    n_cols = len(sent_pos_norm)

    best_scores = np.full((n_rows, n_cols), float('inf'))
    best_indx = np.full((n_rows, n_cols), -1)
    def get_score(r, c):
        nonlocal best_scores
        nonlocal best_indx
        if r < 0:
            return 0
        elif best_scores[r, c] < float('inf'):
            return best_scores[r, c]
        elif r > c:
            return float('inf')
        elif cnk_pos_norm[r] < sent_pos_norm[c]:
            prev_score = get_score(r, c-1)
            if best_scores[r, c] < float('inf'):
                return best_scores[r, c]
            curr_score = (sent_pos_norm[c] - cnk_pos_norm[r]) + get_score(r-1, c-1)
            if prev_score < curr_score:
                best_scores[r, c:] = prev_score
                best_indx[r, c:] =  best_indx[r, c-1]
            else:
                best_scores[r, c] = curr_score
                best_indx[r, c] = c
            return best_scores[r, c]
        else:
            best_scores[r, c] = (cnk_pos_norm[r] - sent_pos_norm[c])  + get_score(r-1, c-1)
            best_indx[r, c] = c
            return best_scores[r, c]
    c_idx = 0
    for r in range(n_rows):
        chunk_pos = cnk_pos_norm[r]
        should_cont = (c_idx < n_cols-1) and (sent_pos_norm[c_idx] < chunk_pos)
        while should_cont:
            c_idx += 1
            should_cont = (c_idx < n_cols-1) and (sent_pos_norm[c_idx] < chunk_pos)
        col_idx = max(0, c_idx - 1)
        while best_scores[r,-1] == float('inf'):
            get_score(r, col_idx)
            col_idx += 1
    return best_scores, best_indx

def get_chunk_ranges(best_indx):
    n_rows, n_cols = best_indx.shape
    r_end = n_cols
    if n_rows == 0:
        return [(0, r_end+1)]
    r_start = best_indx[-1, -1]
    res_breakdown = [(r_start+1, r_end+1)]
    for r in range(n_rows-2, -1, -1):
        r_end = r_start
        r_start = best_indx[r, r_end-1]
        res_breakdown.append((r_start+1, r_end+1))
    res_breakdown.append((0, r_start+1))
    res_breakdown.reverse()
    return res_breakdown

def produce_segments(chapter_token_cnts, n_chunks, min_tok_len, max_tok_len):
    chapter_tok_cnt = sum(chapter_token_cnts)
    mean_tok_len = chapter_tok_cnt / n_chunks
    cl_distr = np.minimum(
        max_tok_len,
        min_tok_len + (mean_tok_len - min_tok_len) * np.random.gamma(shape = 1.5, scale=(1/1.5) , size = n_chunks)
    )
    cnk_pos = list(accumulate(cl_distr))
    chapter_tok_pos = list(accumulate(chapter_token_cnts))
    cnk_pos_norm = (cnk_pos / cnk_pos[-1])[:-1]
    sent_pos_norm = (chapter_tok_pos / np.float64(chapter_tok_pos[-1]))[:-1]

    return get_chunk_ranges(get_scores_and_indx(cnk_pos_norm, sent_pos_norm)[1])

class ChapterDataset(torch.utils.data.Dataset):
    def __init__(
        self, chapter_df, tokenizer, source_cols,
        min_tok_len, avg_tok_len, max_tok_len, max_length,
        model_type, target_col = 'text_ru'
    ):
        self.target_col = target_col
        self.texts_target = chapter_df[self.target_col].tolist()
        present_source_cols = [c for c in source_cols if (chapter_df[c] != '').any()]
        self.texts_source = {
            col_name: chapter_df[col_name].tolist()
            for col_name in present_source_cols
        }
        self.author_tag = a2tag(chapter_df.author.iloc[0])
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
        if model_type == 'T5':
            self.label = tokenizer.encode(self.author_tag)[:-1]
        else:
            self.label = tokenizer.encode(' ' + self.author_tag + ' ')
        self.tot_tokens = sum(self.token_counts)
        self.n_chunks = min(
            max(1,round(self.tot_tokens / avg_tok_len)),
            chapter_df.shape[0]
        )
        self.sample()
        self.eos_token_id =  tokenizer.eos_token_id

    def build_input(self, source_ids, target_ids):
        if self.model_type == 'T5':
            source_ids = self.label + source_ids
            if len(source_ids) > (self.max_length-1):
                source_ids = source_ids[:self.max_length-1]
            if len(target_ids) > (self.max_length-1):
                target_ids = target_ids[:self.max_length-1]
    
            source_ids.append(self.eos_token_id)
            target_ids.append(self.eos_token_id)
    
            attention_mask = [1] * len(source_ids)
    
            return {
                'input_ids': source_ids,
                'attention_mask': attention_mask,
                'labels': target_ids
            }
        elif self.model_type == 'GPT':
            if len(source_ids) > (self.max_length-2):
                source_ids = source_ids[:self.max_length]
            if len(target_ids) > (self.max_length-2):
                target_ids = target_ids[:self.max_length-1]
            input_ids = source_ids + self.label + target_ids + [self.eos_token_id]
            attention_mask = [1] * len(input_ids)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
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
