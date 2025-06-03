from tst_utils.eval.model_names import PERPL_MODEL_NAME
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def calculate_perplexity(model_output, aggregate=False, sep='\n'):
    """ Calculate average perplexity per token and number of tokens in each text."""
    model = AutoModelForCausalLM.from_pretrained(PERPL_MODEL_NAME).cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(PERPL_MODEL_NAME)
    lls = []
    weights = []
    for text in model_output:
        encodings = tokenizer(f'{sep}{text}{sep}', return_tensors='pt')
        input_ids = encodings.input_ids.to(model.device)
        target_ids = input_ids.clone()

        w = max(0, len(input_ids[0]) - 1)
        if w > 0:
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs.loss #outputs[0]
                ll = log_likelihood.item()
        else:
            ll = 0
        lls.append(ll)
        weights.append(w)
    likelihoods, weights = np.array(lls), np.array(weights)
    if aggregate:
        return sum(likelihoods * weights) / sum(weights)
    return likelihoods, weights

def naturality_score(source_perplexity, target_perplexity):
    perpl_scaled_abs = np.maximum(target_perplexity - 4,0)
    perpl_scaled_rel = np.maximum(
        target_perplexity - np.maximum(source_perplexity, 4),0
    )
    perplexity_score_abs = 1/(8 * perpl_scaled_abs + 1)
    perplexity_score_rel = 1/(20 * perpl_scaled_rel + 1)
    return np.sqrt(
        perplexity_score_abs * perplexity_score_rel
    )