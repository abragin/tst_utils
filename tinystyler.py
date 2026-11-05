import hashlib
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM


_VALID_ASSERT_NORM = {"normalized", "unnormalized", None}

# Task 5.2, requirement 6: the registry of checkpoints with a verified
# inference provenance, keyed by the sha256 of the safetensors file.
CHECKPOINT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__),
                                        "tinystyler_checkpoints.json")
# The assert_norm value that matches each registered trained scale.
SCALE_TO_ASSERT_NORM = {"unit": "normalized", "raw": "unnormalized"}


class TinyStylerProvenanceError(ValueError):
    """A TinyStyler checkpoint or its conditioning fails the provenance guard."""


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_checkpoint(sha256):
    """The registry entry of a checkpoint file's sha256, or None."""
    with open(CHECKPOINT_REGISTRY_PATH) as handle:
        registry = json.load(handle)
    entry = registry["checkpoints"].get(sha256)
    if entry is None:
        return None
    entry = dict(entry)
    entry["sha256"] = sha256
    return entry


def _check_entry_against_construction(entry, *, model_name, model_type,
                                      ctrl_embed_dim, assert_norm,
                                      auto_normalize_input):
    for field, value in (("model_name", model_name), ("model_type", model_type),
                         ("ctrl_embed_dim", ctrl_embed_dim)):
        if entry[field] != value:
            raise TinyStylerProvenanceError(
                f"checkpoint {entry['sha256'][:12]} ({entry['name']}) is "
                f"registered with {field}={entry[field]!r}, constructed with "
                f"{value!r}"
            )
    expected = SCALE_TO_ASSERT_NORM[entry["trained_scale"]]
    if assert_norm not in (None, expected):
        raise TinyStylerProvenanceError(
            f"checkpoint {entry['sha256'][:12]} ({entry['name']}) was trained "
            f"on {entry['trained_scale']} style vectors; assert_norm="
            f"{assert_norm!r} contradicts it (pass {expected!r} or None)"
        )
    if auto_normalize_input and entry["trained_scale"] == "raw":
        raise TinyStylerProvenanceError(
            f"checkpoint {entry['sha256'][:12]} ({entry['name']}) is "
            "raw-trained; auto_normalize_input=True would feed it unit vectors"
        )


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
        self._verified_identity = None
        entry = None
        if use_style and checkpoint_path is not None:
            # Hash first, so a contradiction raises before any model load.
            entry = registered_checkpoint(checkpoint_sha256(checkpoint_path))
            if entry is not None:
                _check_entry_against_construction(
                    entry, model_name=model_name, model_type=model_type,
                    ctrl_embed_dim=ctrl_embed_dim, assert_norm=assert_norm,
                    auto_normalize_input=auto_normalize_input)
                # assert_norm=None still enforces the registered scale.
                self.assert_norm = SCALE_TO_ASSERT_NORM[entry["trained_scale"]]
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
            missing, unexpected = torch.nn.Module.load_state_dict(
                self, state_dict, strict=False)
            assert not unexpected, unexpected
            if entry is not None:
                self._verify_loaded_state(entry, set(state_dict), missing)
                # Set only after __init__'s own load: any later
                # load_state_dict clears it.
                self._verified_identity = {
                    key: entry[key] for key in (
                        "sha256", "name", "encoder_key", "trained_scale",
                        "model_name", "model_type", "ctrl_embed_dim")
                }

    def _verify_loaded_state(self, entry, loaded_keys, missing):
        """No missing learned state: every missing key is a registered tied
        alias, tied to a key the file holds (requirement 6)."""
        allowed = {item["key"]: item for item in entry["missing_keys"]}
        if set(missing) != set(allowed):
            raise TinyStylerProvenanceError(
                f"checkpoint {entry['sha256'][:12]} ({entry['name']}): missing "
                f"keys {sorted(missing)}, registered {sorted(allowed)}"
            )
        for key, item in allowed.items():
            if item["kind"] != "tied_alias":
                raise TinyStylerProvenanceError(
                    f"registry kind {item['kind']!r} of {key} is not supported"
                )
            if item["tied_to"] not in loaded_keys:
                raise TinyStylerProvenanceError(
                    f"{key} is tied to {item['tied_to']}, which the file "
                    "does not hold"
                )
            alias = self.get_parameter(key)
            owner = self.get_parameter(item["tied_to"])
            if alias.data_ptr() != owner.data_ptr():
                raise TinyStylerProvenanceError(
                    f"{key} is registered as tied to {item['tied_to']}, but "
                    "the loaded model does not share them"
                )

    @property
    def verified_identity(self):
        """The registry identity set at a verified load, or None (a copy)."""
        identity = getattr(self, "_verified_identity", None)
        return dict(identity) if identity is not None else None

    def load_state_dict(self, *args, **kwargs):
        """Load as torch does; a verified identity does not survive it."""
        self._verified_identity = None
        return super().load_state_dict(*args, **kwargs)

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

    def _conditioning_tensor(self, style, batch_size):
        """The guarded conditioning input of `generate` (requirement 6).

        `style` must be a `StyleSelection`, one row key per input row. The
        checks run right before the projection: the model has a verified
        identity, the selection's encoder is the checkpoint's conditioning
        encoder, and the view follows the registered trained scale. A
        unit-trained checkpoint derives a unit view from any nonzero stored
        values; a raw-trained checkpoint takes the stored raw values and
        rejects unit-only ones, because a raw magnitude cannot be recovered
        from a unit vector. The native-scale contract of the raw encoder
        (norm > 10) is the existing `assert_norm='unnormalized'` check.
        """
        from tst_utils.eval.data.style_store import StyleSelection
        if not isinstance(style, StyleSelection):
            raise TypeError(
                "TinyStyler.generate takes a StyleSelection of labelled style "
                f"vectors, got {type(style).__name__}"
            )
        identity = self.verified_identity
        if identity is None:
            raise TinyStylerProvenanceError(
                "TinyStyler.generate: this model has no verified identity "
                "(an unregistered checkpoint, another load route, or a "
                "load_state_dict after construction)"
            )
        if style.encoder_key != identity["encoder_key"]:
            raise TinyStylerProvenanceError(
                f"TinyStyler.generate: style vectors from "
                f"{style.encoder_key!r}, but checkpoint {identity['name']} is "
                f"conditioned on {identity['encoder_key']!r}"
            )
        if len(style) != batch_size:
            raise ValueError(
                f"{len(style)} style rows for {batch_size} input rows"
            )
        values = np.asarray(style.vectors(), dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        if identity["trained_scale"] == "unit":
            if np.any(norms == 0):
                raise TinyStylerProvenanceError(
                    "a zero style vector has no unit view")
            values = values / norms[:, None]
        else:
            if np.any(np.abs(norms - 1.0) <= 1e-3):
                raise TinyStylerProvenanceError(
                    f"checkpoint {identity['name']} is raw-trained; unit-norm "
                    "style vectors cannot be used (encode with "
                    "normalize=False or declare the raw values)"
                )
        weight = self.proj.weight
        tensor = torch.tensor(values, dtype=weight.dtype, device=weight.device)
        return self._prepare_style(tensor)

    def generate(self, input_ids, attention_mask, style=None, **kwargs):
        """Generate; a style model takes a `StyleSelection` (task 5.2)."""
        if self.use_style:
            style = self._conditioning_tensor(style, input_ids.shape[0])
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
