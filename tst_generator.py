import torch
import numpy as np
from tqdm import tqdm

# --- Predefined generation option sets ---
GEN_OPTS_V1 = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

GEN_OPTS_V2 = {   # beam search
    "num_beams": 4,
    "early_stopping": True,
}

GEN_OPTS_V3 = {   # default TinyStyler sampling params
    "do_sample": True,
    "top_p": 0.8,
    "temperature": 1.0,
}

class TSTGenerator:
    """Batched style-transfer inference wrapper.

    Style-embedding scale: this wrapper passes ``style_emb_dict`` /
    ``target_style_embeddings`` straight through to ``model.generate(style=...)``.
    The caller is responsible for matching the scale to what ``model``
    (typically a :class:`TinyStyler`) expects. Two ways to enforce the
    scale check:

    - construct the model with ``TinyStyler(assert_norm=...)``; or
    - pass ``assert_norm=`` to this wrapper, which will set the same attribute
      on the inner model (overrides whatever the model was constructed with).

    Length parameters (both required, no default):

    - ``max_input_length`` — tokens kept from the input (tokenizer truncation
      budget). Decoupled from the generation budget.
    - ``max_output_length`` — generated-token budget. Maps to
      ``max_new_tokens`` for GPT (decoder-only) and to the decoder
      ``max_length`` for T5 (encoder-decoder).
    - ``min_output_length`` — generated-token floor (``min_new_tokens`` for
      GPT, ``min_length`` for T5).

    Train -> inference relationship: a model trained with per-side
    ``max_side_length`` concatenates source+target for GPT, so a reasonable
    inference setting is ``max_input_length = max_output_length ~ 2 *
    max_side_length`` (short model: 128/side -> 256; long model: 512/side ->
    1024). The previous single ``max_length`` (GPT input truncated to a
    hardcoded ``0.6 * max_length``, then used as the *total* generate budget)
    is removed — that coupling and the ``0.6`` heuristic were undocumented and
    a source of the author-tags truncation bug.
    """
    def __init__(
        self,
        model,
        tokenizer,
        target_styles,
        model_type="T5",  # "T5" or "GPT"
        style_emb_dict=None,  # if None → use author tags
        batch_size=8,
        num_sequences=1,
        generate_options=GEN_OPTS_V1,
        max_input_length=None,   # tokens kept from the input (truncation budget)
        max_output_length=None,  # generated tokens (max_new_tokens for GPT, decoder max_length for T5)
        min_output_length=32,    # generated-token floor (min_new_tokens for GPT, min_length for T5)
        assert_norm=None,
        *,
        max_length=None,  # split into max_input_length / max_output_length; sentinel for the old name
        min_length=None,  # renamed -> min_output_length; sentinel for the old name
    ):
        if max_length is not None:
            raise ValueError(
                "TSTGenerator: `max_length` was split into `max_input_length` "
                "(input truncation budget) and `max_output_length` (generated "
                "tokens). They are no longer coupled, and there is no default — "
                "pass both explicitly (e.g. short model 128/side training -> "
                "max_input_length=max_output_length=256; long model 512/side -> "
                "1024)."
            )
        if min_length is not None:
            raise ValueError(
                "TSTGenerator: `min_length` was renamed to `min_output_length` "
                "(generated-token floor). Pass min_output_length=... instead."
            )
        if max_input_length is None or max_output_length is None:
            raise ValueError(
                "TSTGenerator: `max_input_length` and `max_output_length` are "
                "both required (no default). `max_input_length` truncates the "
                "input; `max_output_length` is the generated-token budget "
                "(max_new_tokens for GPT). Recommended ~2x the training "
                "`max_side_length` per side."
            )

        self.model = model
        self.tokenizer = tokenizer
        self.target_styles = target_styles
        self.model_type = model_type.upper()
        self.style_emb_dict = style_emb_dict
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.generate_options = generate_options
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length

        if assert_norm is not None:
            # Late-override of TinyStyler's scale assertion. Validate here so a
            # bad value fails at wrapper construction, not at first forward().
            from tst_utils.tinystyler import _VALID_ASSERT_NORM
            if assert_norm not in _VALID_ASSERT_NORM:
                raise ValueError(
                    f"assert_norm must be one of {_VALID_ASSERT_NORM}, "
                    f"got {assert_norm!r}"
                )
            if not hasattr(model, 'assert_norm'):
                raise TypeError(
                    "TSTGenerator(assert_norm=...) requires a model that exposes "
                    "an `assert_norm` attribute (e.g. TinyStyler). Got "
                    f"{type(model).__name__}."
                )
            model.assert_norm = assert_norm

        self.device = next(model.parameters()).device

    def _get_style_representation(self, target_style, batch_size):
        """Return style embeddings or author tag string depending on setup."""
        if self.style_emb_dict is not None:  # TinyStyler with embeddings
            if target_style not in self.style_emb_dict:
                raise ValueError(f"Unsupported style: {target_style}")
            emb = self.style_emb_dict[target_style]
            emb = np.array([emb] * batch_size)
            return torch.from_numpy(emb).float().to(self.device)
        else:  # use author tags
            if target_style not in self.target_styles:
                raise ValueError(f"Unsupported style: {target_style}")
            return f"<{target_style.lower()}>"

    def perform_tst(self, texts, target_style=None, target_style_embeddings=None):
        """
        Perform text style transfer on a list of input texts.
        Either `target_style` (str) or `target_style_embeddings` (list-like of embeddings) must be provided.
        Returns a list of generated strings.
        """
        if (target_style is None) and (target_style_embeddings is None):
            raise ValueError("Either target_style or target_style_embeddings must be provided.")

        if (target_style is not None) and (target_style_embeddings is not None):
            raise ValueError("Provide only one of target_style or target_style_embeddings, not both.")

        results = []
        total = len(texts)

        for start in tqdm(range(0, total, self.batch_size)):
            end = start + self.batch_size
            batch_texts = texts[start:end]

            # --- Determine style embeddings or tag ---
            if target_style_embeddings is not None:
                batch_embs = target_style_embeddings[start:end]
                if len(batch_embs) != len(batch_texts):
                    raise ValueError("Length mismatch between texts and target_style_embeddings.")
                style_embs = torch.tensor(np.stack(batch_embs)).float().to(self.device)
            elif self.style_emb_dict is not None:
                style_embs = self._get_style_representation(target_style, len(batch_texts))
            else:
                style_embs = None # will use author tags path

            if style_embs is not None:
                # embeddings path
                if self.model_type == "GPT":
                    ipt = self.tokenizer(
                        [t + self.tokenizer.eos_token for t in batch_texts],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    )
                    gen_len_kwargs = {
                        "max_new_tokens": self.max_output_length,
                        "min_new_tokens": self.min_output_length,
                    }
                else:  # T5
                    ipt = self.tokenizer(
                        list(batch_texts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    )
                    gen_len_kwargs = {
                        "max_length": self.max_output_length,
                        "min_length": self.min_output_length,
                    }

                ipt = {k: v.to(self.device) for k, v in ipt.items()}

                # Generate
                output_sequences = self.model.generate(
                    input_ids=ipt["input_ids"],
                    attention_mask=ipt["attention_mask"],
                    style=style_embs,
                    num_return_sequences=self.num_sequences,
                    **gen_len_kwargs,
                    **self.generate_options,
                )
                decoded = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            else:  # author tags path
                style_token = self._get_style_representation(target_style, len(batch_texts))

                if self.model_type == "GPT":
                    # The style tag goes at the END of GPT inputs and the
                    # post-generation cleanup splits on it. Truncate the text
                    # first and append the tag AFTER, so truncation can never
                    # clip the trailing tag.
                    tag_ids = self.tokenizer(
                        " " + style_token, add_special_tokens=False
                    )["input_ids"]
                    keep = max(1, self.max_input_length - len(tag_ids))
                    batch_input_ids = [
                        self.tokenizer(
                            txt, add_special_tokens=False,
                            truncation=True, max_length=keep,
                        )["input_ids"] + tag_ids
                        for txt in batch_texts
                    ]
                    ipt = self.tokenizer.pad(
                        {"input_ids": batch_input_ids},
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                    gen_len_kwargs = {
                        "max_new_tokens": self.max_output_length,
                        "min_new_tokens": self.min_output_length,
                    }
                else:  # T5 — tag is at the front, simple truncation is safe
                    inputs = [style_token + " " + txt for txt in batch_texts]
                    ipt = self.tokenizer(
                        inputs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    ).to(self.device)
                    gen_len_kwargs = {
                        "max_length": self.max_output_length,
                        "min_length": self.min_output_length,
                    }

                output_ids = self.model.generate(
                    **ipt,
                    num_return_sequences=self.num_sequences,
                    **gen_len_kwargs,
                    **self.generate_options,
                )

                # For GPT with tags, strip everything before/including tag if needed
                if self.model_type == "GPT":
                    decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
                    cleaned = []
                    for t in decoded:
                        if style_token in t:
                            t = t.split(style_token, 1)[1].split(self.tokenizer.eos_token)[0]
                        cleaned.append(t.strip())
                    decoded = cleaned
                else:
                    decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            grouped = [decoded[i:i + self.num_sequences] for i in range(0, len(decoded), self.num_sequences)]
            results.extend(grouped)

        return results
