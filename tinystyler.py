from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM


_VALID_ASSERT_NORM = {"normalized", "unnormalized", None}


class TinyStyler(torch.nn.Module):
    """T5/GPT wrapper that prepends a projected style embedding to inputs.

    Style-embedding scale guards (see docs/issues/resolved/style-embedding-normalization.md):

    - ``assert_norm='normalized'`` (default): assert per-row style norm is
      ~1.0 (folder-14 and later canonical scale). Loading older checkpoints
      (folders 9/11) with their unnormalized centroids will hard-fail; pass
      ``assert_norm='unnormalized'`` in that case.
    - ``assert_norm='unnormalized'``: assert per-row style norm > 10
      (pre-folder-14 centroids; norm typically ~15).
    - ``assert_norm=None``: no assertion (use when scale is heterogeneous or
      already validated upstream).
    - ``auto_normalize_input=True``: L2-normalize the incoming style vector
      before projection. Useful when feeding pre-folder-14 centroids to a
      folder-14 checkpoint without modifying the source data.
    """

    def __init__(
        self, model_name,
        model_type='T5', use_style=False, ctrl_embed_dim=768,
        checkpoint_path=None,
        auto_normalize_input: bool = False,
        assert_norm: Optional[str] = "normalized",
    ):
        super().__init__()
        if model_type in ['T5', 'GPT']:
            self.model_type = model_type
        else:
            raise Exception("Unsupported model type: ", model_type)
        if assert_norm not in _VALID_ASSERT_NORM:
            raise ValueError(
                f"assert_norm must be one of {_VALID_ASSERT_NORM}, "
                f"got {assert_norm!r}"
            )
        if auto_normalize_input and assert_norm == "unnormalized":
            # The norm assertion runs after auto-normalization, so it would
            # always fail (auto-normalized vectors have norm 1.0, not > 10).
            # Catch the contradiction at construction time.
            raise ValueError(
                "auto_normalize_input=True is incompatible with "
                "assert_norm='unnormalized': auto-normalization happens before "
                "the assertion, so the assertion would always fail. Pick one."
            )
        self.auto_normalize_input = auto_normalize_input
        self.assert_norm = assert_norm
        if model_type == 'T5':
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                use_safetensors=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_safetensors=True
            )
        self.use_style = use_style
        if self.use_style:
            self.ctrl_embed_dim = ctrl_embed_dim
            if hasattr(self.model.config, 'd_model'):
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.d_model
                )
            elif hasattr(self.model.config, 'n_embd'):
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.n_embd
                )
            else:
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.hidden_size
                )
        if checkpoint_path is not None:
            state_dict = load_file(checkpoint_path)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            assert not unexpected, unexpected

    def _prepare_style(self, style: torch.Tensor) -> torch.Tensor:
        """Apply optional auto-normalization and per-row norm assertion.

        Expects ``style`` of shape ``(batch, ctrl_embed_dim)``.
        """
        if self.auto_normalize_input:
            style = F.normalize(style, p=2, dim=-1)
        if self.assert_norm is not None:
            norms = style.detach().norm(p=2, dim=-1)
            if self.assert_norm == "normalized":
                ok = ((norms >= 0.9) & (norms <= 1.1)).all().item()
                if not ok:
                    raise AssertionError(
                        f"TinyStyler(assert_norm='normalized') expected unit-norm "
                        f"style vectors (per-row ||.|| in [0.9, 1.1]), got norms "
                        f"min={norms.min().item():.4f} max={norms.max().item():.4f}. "
                        f"If you are using a pre-folder-14 centroid (typical norm "
                        f"~15), construct TinyStyler(assert_norm='unnormalized') "
                        f"or set auto_normalize_input=True."
                    )
            elif self.assert_norm == "unnormalized":
                ok = (norms > 10).all().item()
                if not ok:
                    raise AssertionError(
                        f"TinyStyler(assert_norm='unnormalized') expected raw "
                        f"encoder-scale style vectors (per-row ||.|| > 10), got "
                        f"norms min={norms.min().item():.4f} "
                        f"max={norms.max().item():.4f}. If you are using "
                        f"folder-14 unit-norm centroids, construct "
                        f"TinyStyler(assert_norm='normalized')."
                    )
        return style

    def forward(self, input_ids, attention_mask, labels=None, style=None):
        if self.use_style:
            style = self._prepare_style(style)
            style_embed = self.proj(style).unsqueeze(1)

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )
            if (self.model_type == 'GPT') and (labels is not None):
                # pad labels for extra token
                pad_label = torch.full((labels.size(0), 1), -100, device=labels.device)
                labels = torch.cat([pad_label, labels], dim=1)

        return self.model(
            inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels
        )

    def generate(self, input_ids, attention_mask, style=None, **kwargs):
        if self.use_style:
            style = self._prepare_style(style)
            # Note: projection order differs from forward() — forward() does
            # self.proj(style).unsqueeze(1), this does self.proj(style.unsqueeze(1)).
            # Both produce shape (B, 1, d_model) — Linear broadcasts over the
            # added singleton dim. Kept asymmetric for historical compatibility
            # with existing checkpoints; do not "harmonize" without retesting.
            style_embed = self.proj(style.unsqueeze(1))

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        return self.model.generate(
            inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs
        )
