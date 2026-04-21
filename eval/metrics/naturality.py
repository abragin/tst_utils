from tst_utils.eval.model_names import PERPL_MODEL_NAME
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def calculate_perplexity(model_output, aggregate=False, sep='\n', batch_size=32):
    """Compute average per-token CE loss for each text using rugpt3small.

    Returns (likelihoods, weights) where likelihoods[i] is mean CE loss for text i
    and weights[i] is the token count. If aggregate=True returns a single
    weighted-average CE loss over the full list.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        PERPL_MODEL_NAME,
        use_safetensors=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(PERPL_MODEL_NAME)

    texts = [f'{sep}{t}{sep}' for t in model_output]

    lls = []
    ws = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', truncation=True,
                        max_length=512, padding=True).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc['input_ids'])
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = enc['input_ids'][..., 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(
                reduction='none', ignore_index=tokenizer.pad_token_id
            )
            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            attn = enc['attention_mask'][..., 1:].float()
            sample_loss = (token_losses * attn).sum(-1) / attn.sum(-1).clamp(min=1)
            sample_w = attn.sum(-1).long()
        lls.extend(sample_loss.cpu().numpy().tolist())
        ws.extend(sample_w.cpu().numpy().tolist())

    likelihoods, weights = np.array(lls), np.array(ws)
    if aggregate:
        return (likelihoods * weights).sum() / weights.sum()
    return likelihoods, weights


def naturality_score(source_perplexity, target_perplexity):
    perpl_scaled_abs = np.maximum(target_perplexity - 4, 0)
    perpl_scaled_rel = np.maximum(
        target_perplexity - np.maximum(source_perplexity, 4), 0
    )
    perplexity_score_abs = 1 / (8 * perpl_scaled_abs + 1)
    perplexity_score_rel = 1 / (20 * perpl_scaled_rel + 1)
    return np.sqrt(perplexity_score_abs * perplexity_score_rel)


