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

    Style conditioning (task 5.2): ``style_emb_dict`` must be
    ``LabelledStyleVectors`` whose encoder is the conditioning encoder of the
    model's verified identity (a registered TinyStyler checkpoint), checked at
    construction; ``target_style_embeddings`` must be a ``StyleSelection``.
    The wrapper passes selections to ``model.generate(style=...)``, and
    TinyStyler builds the tensor at the registered trained scale. Bare
    arrays and dicts raise. ``assert_norm=`` on a verified model must equal
    the registered scale; on another model it sets the attribute, as before.
    Author-tag models (``style_emb_dict=None``) are unchanged.

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
        identity = getattr(model, "verified_identity", None)
        if style_emb_dict is not None:
            # Task 5.2: the named-target conditioning route takes only a
            # labelled object, checked against the model's verified identity
            # here, at construction.
            from tst_utils.eval.data.style_store import LabelledStyleVectors
            if not isinstance(style_emb_dict, LabelledStyleVectors):
                raise TypeError(
                    "TSTGenerator(style_emb_dict=...) takes LabelledStyleVectors "
                    "(load_labelled_author_styles, from_encoded or declare), "
                    f"got {type(style_emb_dict).__name__}"
                )
            if identity is None:
                raise ValueError(
                    "TSTGenerator(style_emb_dict=...) needs a model with a "
                    "verified identity (a registered TinyStyler checkpoint)"
                )
            if style_emb_dict.encoder_key != identity["encoder_key"]:
                raise ValueError(
                    f"style_emb_dict holds {style_emb_dict.encoder_key!r} "
                    f"vectors; the checkpoint {identity['name']} is conditioned "
                    f"on {identity['encoder_key']!r}"
                )
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
            if identity is not None:
                from tst_utils.tinystyler import SCALE_TO_ASSERT_NORM
                registered = SCALE_TO_ASSERT_NORM[identity["trained_scale"]]
                if assert_norm != registered:
                    raise ValueError(
                        f"TSTGenerator(assert_norm={assert_norm!r}) contradicts "
                        f"the registered trained scale of {identity['name']} "
                        f"({identity['trained_scale']}, assert_norm="
                        f"{registered!r})"
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
            # A selection of the labelled source; TinyStyler.generate checks
            # it and builds the tensor right before the projection.
            return self.style_emb_dict.select([target_style] * batch_size)
        else:  # use author tags
            if target_style not in self.target_styles:
                raise ValueError(f"Unsupported style: {target_style}")
            return f"<{target_style.lower()}>"

    def _gen_len_kwargs(self):
        """Generated-length kwargs, keyed by model family.

        GPT (decoder-only) counts *new* tokens; T5 (encoder-decoder) counts
        decoder ``max_length`` / ``min_length``. See the class docstring.
        """
        if self.model_type == "GPT":
            return {
                "max_new_tokens": self.max_output_length,
                "min_new_tokens": self.min_output_length,
            }
        return {
            "max_length": self.max_output_length,
            "min_length": self.min_output_length,
        }

    def _prepare_inputs(self, batch_texts, style_token=None):
        """Tokenize a batch into model-ready, on-device generate inputs.

        ``style_token=None`` -> embeddings path (style is supplied separately
        via the ``style=`` generate kwarg). ``style_token`` set -> author-tag
        path, where the tag is woven into the text itself.
        """
        # GPT pads on the LEFT, passed per call. A decoder-only model continues
        # from the last position, so right pads would sit between the prompt
        # and the continuation and degrade the output with no error. Do not
        # rely on the tokenizer default: rugpt3small ships 'left', but
        # eval/metrics/naturality.py sets its own instance of the same base
        # model's tokenizer to 'right' on purpose (batch-invariant CE). A
        # per-call argument leaves the caller's tokenizer unmutated. See
        # docs/issues/resolved/tstgenerator-gpt-padding-side.md.
        # T5 branches: an encoder-decoder attends to the whole input, so the
        # side does not matter there.
        if style_token is None:
            # Embeddings path: GPT gets a trailing eos; T5 takes the raw text.
            if self.model_type == "GPT":
                ipt = self.tokenizer(
                    [t + self.tokenizer.eos_token for t in batch_texts],
                    return_tensors="pt", padding=True, truncation=True,
                    max_length=self.max_input_length, padding_side="left",
                )
            else:  # T5
                ipt = self.tokenizer(
                    list(batch_texts),
                    return_tensors="pt", padding=True, truncation=True,
                    max_length=self.max_input_length,
                )
            return {k: v.to(self.device) for k, v in ipt.items()}

        # Author-tag path.
        if self.model_type == "GPT":
            # The tag goes at the END of GPT inputs and the post-generation
            # cleanup splits on it. Truncate the text first and append the tag
            # AFTER, so truncation can never clip the trailing tag.
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
                {"input_ids": batch_input_ids}, padding=True, return_tensors="pt",
                padding_side="left",
            )
        else:  # T5 — tag is at the front, simple truncation is safe
            inputs = [style_token + " " + txt for txt in batch_texts]
            ipt = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_input_length,
            )
        return ipt.to(self.device)

    def _decode(self, output_ids, style_token=None):
        """Decode generated ids to strings.

        Only the tag+GPT path needs special handling: those outputs include the
        prompt (generated from ``input_ids``), so we split on the trailing tag
        and trim at the eos. Every other path — including embeddings+GPT, which
        generates from ``inputs_embeds`` and therefore returns *only* new tokens
        (no prompt echo) — decodes plainly with ``skip_special_tokens=True``.
        """
        if style_token is not None and self.model_type == "GPT":
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
            cleaned = []
            for t in decoded:
                if style_token in t:
                    t = t.split(style_token, 1)[1].split(self.tokenizer.eos_token)[0]
                cleaned.append(t.strip())
            return cleaned
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

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

        if target_style_embeddings is not None:
            from tst_utils.eval.data.style_store import StyleSelection
            if not isinstance(target_style_embeddings, StyleSelection):
                raise TypeError(
                    "perform_tst(target_style_embeddings=...) takes a "
                    "StyleSelection (a labelled source and one row key per "
                    f"text), got {type(target_style_embeddings).__name__}"
                )
            if len(target_style_embeddings) != len(texts):
                raise ValueError(
                    "Length mismatch between texts and target_style_embeddings.")
            identity = getattr(self.model, "verified_identity", None)
            if (identity is not None and target_style_embeddings.encoder_key
                    != identity["encoder_key"]):
                raise ValueError(
                    f"target style vectors from "
                    f"{target_style_embeddings.encoder_key!r}; the checkpoint "
                    f"{identity['name']} is conditioned on "
                    f"{identity['encoder_key']!r}"
                )

        results = []
        total = len(texts)

        for start in tqdm(range(0, total, self.batch_size)):
            end = start + self.batch_size
            batch_texts = texts[start:end]

            # Resolve this batch's style representation: either an embeddings
            # tensor (passed via the `style=` generate kwarg) or an author-tag
            # string (woven into the input text by `_prepare_inputs`).
            style_embs, style_token = None, None
            if target_style_embeddings is not None:
                style_embs = target_style_embeddings.source.select(
                    target_style_embeddings.row_keys[start:end])
            elif self.style_emb_dict is not None:
                style_embs = self._get_style_representation(target_style, len(batch_texts))
            else:
                style_token = self._get_style_representation(target_style, len(batch_texts))

            ipt = self._prepare_inputs(batch_texts, style_token=style_token)
            gen_kwargs = {
                "num_return_sequences": self.num_sequences,
                **self._gen_len_kwargs(),
            }
            if style_embs is not None:
                gen_kwargs["style"] = style_embs

            # Fail fast if generate_options tries to set a wrapper-controlled
            # kwarg. Pre-refactor these were passed explicitly before
            # **self.generate_options, so a duplicate raised TypeError; merging
            # into one dict would instead let generate_options silently win.
            # Key on the *active* kwargs only (e.g. GPT reserves max_new_tokens,
            # not max_length), matching the original collision semantics.
            clash = (set(gen_kwargs) | set(ipt)) & set(self.generate_options)
            if clash:
                raise ValueError(
                    f"generate_options may not set wrapper-controlled kwargs "
                    f"{sorted(clash)} — these come from num_sequences / "
                    f"max_output_length / min_output_length / style / the "
                    f"tokenized inputs. Remove them from generate_options."
                )

            output_ids = self.model.generate(**ipt, **gen_kwargs, **self.generate_options)
            decoded = self._decode(output_ids, style_token=style_token)

            grouped = [decoded[i:i + self.num_sequences] for i in range(0, len(decoded), self.num_sequences)]
            results.extend(grouped)

        return results
