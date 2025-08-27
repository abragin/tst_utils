import numpy as np
from itertools import accumulate


def gpt_tokenize_function(batch, tokenizer, max_length=256):
    # Build concatenated texts with EOS tokens
    input_txts = [
        src + tokenizer.eos_token + tgt + tokenizer.eos_token
        for src, tgt in zip(batch["source"], batch["target"])
    ]

    # Tokenize the whole batch
    model_inputs = tokenizer(
        input_txts,
        max_length=max_length,
        padding=False,
        truncation=True,
    )

    return model_inputs

def t5_tokenize_function(example, tokenizer, max_length = 128):
     return tokenizer(
        example["source"],
        text_target = example["target"],
        max_length=max_length,
        padding=False,
        truncation=True
    )

def processing_fn(ds, tokenizer, max_length, model_type="T5"):
    if model_type == "T5":
        tok_fn = t5_tokenize_function
    elif model_type == "GPT":
        tok_fn = gpt_tokenize_function
    else:
        raise Exception("Unsupported model type: ", model_type)
    return ds.select_columns(
        ['source', 'target', 'target_style_emb']
    ).rename_column(
        "target_style_emb", "style"
    ).map(
        lambda batch: tok_fn(batch, tokenizer, max_length=max_length),
        batched=True
    ).remove_columns(['source', 'target'])

def a2tag(author):
    return f"<{author.lower().replace(' ', '_')}>"

def print_item(inputs, tokenizer):
    source = tokenizer.decode(inputs['input_ids'])
    print(source)
    if 'labels' in inputs:
        target = tokenizer.decode(inputs['labels'])
        print(target)

def print_item_with_debug(item, tokenizer):
    inputs = item[0]
    db_info = item[1]
    print_item(inputs, tokenizer)
    print(db_info)

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