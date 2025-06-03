from datasets import Dataset
import numpy as np
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

def labse_meaning_score(text, styled_text):
    model = AutoModel.from_pretrained(LABSE_MODEL_NAME).cuda()
    tokenizer = AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
    model.eval()
    embs_input = labse_embeddings(text, model, tokenizer)
    embs_output = labse_embeddings(styled_text, model, tokenizer)
    scores = np.array([
        torch.dot(embs_input[i], embs_output[i]).cpu().numpy().item()
        for i in range(len(embs_input))
    ])
    return scores

def b_score(text, styled_text):
    """Compute BERTScore F1 for given text pairs."""
    P, R, F1 = bert_score.score(list(text), list(styled_text), lang='ru', verbose=False)
    raw_score = F1.numpy()
    return raw_score

def meaning_score(bs_raw: np.ndarray, ls_raw: np.ndarray) -> np.ndarray:
    """Combine BERTScore and LaBSE score using min-max normalization & geometric mean."""
    return np.sqrt(
        min_max_bert_score(bs_raw) *
        min_max_labse(ls_raw)
    )
