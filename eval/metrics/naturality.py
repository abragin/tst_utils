"""
Naturality: mean per-token cross-entropy loss from rugpt3small.

Model revision — this load stays unpinned, and a test guards it
---------------------------------------------------------------
calculate_perplexity loads PERPL_MODEL_NAME with no revision, by a host decision of
2026-10-06. The reason is that the model loads from two commits at once, so a pin
would change which files the load reads:

  - a9307e696cd3c5b7f953ff4cb19d76a4d81821d5 is refs/main. It holds the config, the
    tokenizer files and pytorch_model.bin. It holds no safetensors file.
  - 0dc3542988c6f6475797ce6d019dce7cb0081e86 holds a config and model.safetensors,
    and no tokenizer file.

With use_safetensors=True the unpinned load takes its config from the first commit
and its weights from the second. Measured on tallin on 2026-10-06, by a patch on
transformers.modeling_utils.load_state_dict, which named
snapshots/0dc3542988.../model.safetensors as the file it opened. A load pinned to
a9307e696c... raises OSError, because no safetensors file exists at that revision.

Two consequences:

  1. The obvious pin value, the hash the unpinned load reports, breaks the load. A
     pin taken from refs/main or from config._commit_hash must be tested by loading
     it.
  2. config._commit_hash names the commit the config resolved at, and says nothing
     about the weights. A test that asserts it proves which config was used, and
     not which weights.

So a pin here would move the config and the tokenizer to 0dc3542988..., which is the
only behaviour change the pinning task would have made, against a golden that scores
six texts in four configurations. Instead test_naturality.py asserts the config
revision. A moved default branch then fails in the test rather than silently in a
score. That assertion needs the Hub to be reachable: with a warm cache and no
network, the load resolves the cached snapshot and the test passes.

See docs/inbox/2026-10-06-unpinned-load-mixes-two-commits.md.
"""

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
    # rugpt3small's tokenizer defaults to padding_side='left'. With left padding and
    # no position_ids, GPT2 assigns real tokens position ids that include the per-batch
    # pad offset, shifting their positional embeddings -> per-text CE becomes
    # batch-composition dependent (up to ~7.6 CE units; see
    # docs/issues/resolved/calculate-perplexity-left-pad-batch-dependence.md). Right padding
    # keeps real tokens at positions 0..n-1 and out of causal attention's view of the
    # pads, making CE batch-invariant (verified: max |bs1-bs8| = 2e-6).
    tokenizer.padding_side = 'right'

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
    """DEPRECATED (superseded by nat_v2 in composite.py). Absolute+relative CE
    naturality score. Retained only for the v1 composite `score` (scoring.py) and
    its golden test; not a generation gate. NOT recalibrated and does not need to be:
    it predates the left-padding CE bug and its thresholds (4 / k_abs 8 / k_rel 20)
    were tuned on the correct bs=1 CE scale, so with calculate_perplexity now fixed
    it operates on the correct scale again (mean corrected CE ~3.76). The args named
    *_perplexity are mean per-token CE, not perplexity. Unlikely to be used going
    forward; kept for backward compatibility. Do not alter the constants here."""
    perpl_scaled_abs = np.maximum(target_perplexity - 4, 0)
    perpl_scaled_rel = np.maximum(
        target_perplexity - np.maximum(source_perplexity, 4), 0
    )
    perplexity_score_abs = 1 / (8 * perpl_scaled_abs + 1)
    perplexity_score_rel = 1 / (20 * perpl_scaled_rel + 1)
    return np.sqrt(perplexity_score_abs * perplexity_score_rel)


