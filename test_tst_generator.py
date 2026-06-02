"""Fake-model tests for TSTGenerator length-parameter semantics.

These use a real HF tokenizer (so truncation behaves realistically) and a
fake model that records the kwargs passed to ``generate`` and echoes the
input ids back as the "generated" sequence. That is enough to prove:

- input is truncated to ``max_input_length`` (not some fraction of it),
- generation is asked for ``max_new_tokens=max_output_length`` (GPT) /
  decoder ``max_length=max_output_length`` (T5) — decoupled from the input,
- the GPT author-tags path keeps the trailing style tag even when the input
  is longer than ``max_input_length`` (the bug this refactor fixes),
- the old ``max_length`` / ``min_length`` kwargs hard-fail with ValueError.
"""
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from tst_utils.tst_generator import TSTGenerator, GEN_OPTS_V1


GPT_NAME = "ai-forever/rugpt3small_based_on_gpt2"
T5_NAME = "ai-forever/ruT5-base"


class FakeModel(torch.nn.Module):
    """Records the last ``generate`` call and echoes the input ids back."""

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.last_call = None

    def generate(self, input_ids=None, attention_mask=None, style=None, **kwargs):
        self.last_call = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "style": style,
            "kwargs": kwargs,
        }
        # Echo input ids as the "generated" sequence (valid ids -> decodable).
        return input_ids


@pytest.fixture(scope="module")
def gpt_tokenizer():
    tok = AutoTokenizer.from_pretrained(GPT_NAME)
    tok.add_special_tokens({"additional_special_tokens": ["<news>"]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def t5_tokenizer():
    tok = AutoTokenizer.from_pretrained(T5_NAME)
    tok.add_special_tokens({"additional_special_tokens": ["<news>"]})
    return tok


LONG_TEXT = "слово " * 200  # safely longer than the small max_input_length below


def test_gpt_embeddings_path_decouples_input_and_output(gpt_tokenizer):
    model = FakeModel()
    style_dim = 8
    gen = TSTGenerator(
        model, gpt_tokenizer, target_styles=None, model_type="GPT",
        style_emb_dict={"News": np.ones(style_dim, dtype=np.float32)},
        batch_size=4, generate_options={},
        max_input_length=16, max_output_length=24, min_output_length=5,
    )
    gen.perform_tst([LONG_TEXT, LONG_TEXT], target_style="News")

    call = model.last_call
    # input truncated to max_input_length (not 0.6 * something)
    assert call["input_ids"].shape[1] <= 16
    # generation budget is in NEW tokens, == max_output_length
    assert call["kwargs"]["max_new_tokens"] == 24
    assert call["kwargs"]["min_new_tokens"] == 5
    assert "max_length" not in call["kwargs"]
    assert call["style"] is not None


def test_gpt_author_tags_keeps_trailing_tag(gpt_tokenizer):
    model = FakeModel()
    gen = TSTGenerator(
        model, gpt_tokenizer, target_styles=["News"], model_type="GPT",
        style_emb_dict=None, batch_size=4, generate_options={},
        max_input_length=16, max_output_length=24, min_output_length=5,
    )
    gen.perform_tst([LONG_TEXT, LONG_TEXT], target_style="News")

    call = model.last_call
    tag_ids = gpt_tokenizer(" <news>", add_special_tokens=False)["input_ids"]
    ids = call["input_ids"]
    # Every row must END with the full style tag despite truncation, otherwise
    # the post-generation split on the tag would silently drop output.
    for row in ids:
        row = row.tolist()
        assert row[-len(tag_ids):] == tag_ids, (
            "style tag was clipped by truncation"
        )
    assert call["kwargs"]["max_new_tokens"] == 24
    assert call["kwargs"]["min_new_tokens"] == 5


def test_t5_uses_decoder_max_length(t5_tokenizer):
    model = FakeModel()
    style_dim = 8
    gen = TSTGenerator(
        model, t5_tokenizer, target_styles=None, model_type="T5",
        style_emb_dict={"News": np.ones(style_dim, dtype=np.float32)},
        batch_size=4, generate_options={},
        max_input_length=16, max_output_length=24, min_output_length=5,
    )
    gen.perform_tst([LONG_TEXT, LONG_TEXT], target_style="News")

    call = model.last_call
    assert call["input_ids"].shape[1] <= 16
    # T5 (encoder-decoder): decoder length is the plain max_length/min_length
    assert call["kwargs"]["max_length"] == 24
    assert call["kwargs"]["min_length"] == 5
    assert "max_new_tokens" not in call["kwargs"]


@pytest.mark.parametrize("bad_opts", [
    {"num_return_sequences": 3},   # collides with self.num_sequences
    {"max_new_tokens": 99},        # collides with the active GPT length kwarg
    {"style": object()},           # collides with the embeddings style kwarg
    {"input_ids": None},           # collides with the tokenized inputs
])
def test_conflicting_generate_options_raises(gpt_tokenizer, bad_opts):
    """generate_options must not override wrapper-controlled generate kwargs.

    Pre-refactor these were passed explicitly before **generate_options, so a
    duplicate raised. The merged-dict refactor must keep failing fast rather
    than letting generate_options silently win.
    """
    model = FakeModel()
    gen = TSTGenerator(
        model, gpt_tokenizer, target_styles=None, model_type="GPT",
        style_emb_dict={"News": np.ones(8, dtype=np.float32)},
        batch_size=4, generate_options=bad_opts,
        max_input_length=16, max_output_length=24, min_output_length=5,
    )
    with pytest.raises(ValueError, match="wrapper-controlled"):
        gen.perform_tst(["слово"], target_style="News")


def test_nonactive_length_key_does_not_clash(gpt_tokenizer):
    """A GPT model reserves max_new_tokens, not max_length — so a stray
    max_length in generate_options is NOT a wrapper collision (matches the
    pre-refactor per-kwarg semantics) and is forwarded to generate as-is."""
    model = FakeModel()
    gen = TSTGenerator(
        model, gpt_tokenizer, target_styles=None, model_type="GPT",
        style_emb_dict={"News": np.ones(8, dtype=np.float32)},
        batch_size=4, generate_options={"max_length": 50},
        max_input_length=16, max_output_length=24, min_output_length=5,
    )
    gen.perform_tst(["слово"], target_style="News")  # must not raise
    assert model.last_call["kwargs"]["max_length"] == 50
    assert model.last_call["kwargs"]["max_new_tokens"] == 24


def test_old_max_length_kwarg_raises(gpt_tokenizer):
    with pytest.raises(ValueError, match="max_input_length"):
        TSTGenerator(
            FakeModel(), gpt_tokenizer, target_styles=["News"], model_type="GPT",
            generate_options=GEN_OPTS_V1, max_length=512,
        )


def test_old_min_length_kwarg_raises(gpt_tokenizer):
    with pytest.raises(ValueError, match="min_output_length"):
        TSTGenerator(
            FakeModel(), gpt_tokenizer, target_styles=["News"], model_type="GPT",
            generate_options=GEN_OPTS_V1,
            max_input_length=256, max_output_length=256, min_length=32,
        )


def test_required_lengths_have_no_default(gpt_tokenizer):
    with pytest.raises(ValueError, match="required"):
        TSTGenerator(
            FakeModel(), gpt_tokenizer, target_styles=["News"], model_type="GPT",
            generate_options=GEN_OPTS_V1,
        )
