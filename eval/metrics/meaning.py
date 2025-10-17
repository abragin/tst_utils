from datasets import Dataset
import numpy as np
import pandas as pd
import bert_score
from tst_utils.eval.model_names import LABSE_MODEL_NAME
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel


LABSE_MIN = 0.2
LABSE_MAX = 1
BERT_SCORE_MIN = 0.45
BERT_SCORE_MAX = 1

def min_max_labse(x):
    norm_score = (x - LABSE_MIN) / (LABSE_MAX - LABSE_MIN)
    return np.clip(norm_score, 0,1)

def min_max_bert_score(x):
    norm_score = (x - BERT_SCORE_MIN) / (BERT_SCORE_MAX - BERT_SCORE_MIN)
    return np.clip(norm_score, 0,1)

def labse_embeddings(texts, labse_model, labse_tokenizer, batch_size=16):
    dataset = Dataset.from_dict({'text': texts})
    tok_func_labse = lambda x: labse_tokenizer(x["text"], truncation=True, max_length=512)
    data_collator_labse = DataCollatorWithPadding(
        tokenizer=labse_tokenizer
    )
    tok_ds = dataset.map(
        tok_func_labse, batched=True, remove_columns=('text',))
    dataloader = DataLoader(
        tok_ds, batch_size = batch_size,
        collate_fn = data_collator_labse
    )
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            # Move the batch to the same device as the model
            batch = {k: v.to('cuda') for k, v in batch.items()}
            # Perform inference
            outputs = labse_model(**batch)
            ems = F.normalize(outputs.pooler_output)
            # Store probabilities
            embeddings.extend(ems)
    return embeddings

def calc_labse_embeddings(texts):
    model = AutoModel.from_pretrained(LABSE_MODEL_NAME).cuda()
    tokenizer = AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
    model.eval()
    return labse_embeddings(texts, model, tokenizer)

def labse_scores_from_embs(text_labse_embs, style_text_labse_embs):
    scores = []
    for a, b in zip(text_labse_embs, style_text_labse_embs):
        # Convert both inputs to CPU tensors safely
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b)

        # Move to CPU
        a = a.cpu()
        b = b.cpu()

        # Compute dot product
        score = torch.dot(a, b).item()
        scores.append(score)

    return np.array(scores)

def length_ratio(texts, styled_texts):
    """
    Compute symmetric length ratio between pairs of texts:
    ratio = min(len_in, len_out) / max(len_in, len_out)

    Works with lists or pandas Series of strings.
    Returns a numpy array of ratios in [0, 1].
    """
    # Convert to pandas Series for consistent vectorization
    s1 = pd.Series(texts, dtype=str)
    s2 = pd.Series(styled_texts, dtype=str)

    len_in = s1.str.len()
    len_out = s2.str.len()

    # Avoid division by zero
    ratio = np.where(
        (len_in + len_out) == 0,
        0.0,
        np.minimum(len_in, len_out) / np.maximum(len_in, len_out)
    )
    return ratio

def sin_penalty(x):
    return 0.5 - 0.5 * np.cos(np.pi * x)

def length_penalty(texts, styled_texts):
    return sin_penalty(
        length_ratio(texts, styled_texts)
    )

def b_score(texts, styled_texts):
    """Compute BERTScore F1 for given text pairs."""
    P, R, F1 = bert_score.score(list(texts), list(styled_texts), lang='ru', verbose=False)
    raw_score = F1.numpy()
    return raw_score

def meaning_score(bs_raw: np.ndarray, ls_raw: np.ndarray, length_penalties: np.ndarray) -> np.ndarray:
    """Combine BERTScore and LaBSE score using min-max normalization & geometric mean."""
    return length_penalties * np.sqrt(
        min_max_bert_score(bs_raw) *
        min_max_labse(ls_raw)
    )
