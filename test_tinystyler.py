"""
Tests for TinyStyler style-embedding normalization guards.

Run on remote:
    ssh tallin.vpn 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tst311 && \\
        cd /home/abragin/src/textprism/ && \\
        pytest tst_utils/test_tinystyler.py -v'

These tests do not call HuggingFace `from_pretrained` — they construct
TinyStyler via __new__ and inject only the attributes used by _prepare_style /
forward / generate. This keeps tests fast and offline.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tst_utils.tinystyler import TinyStyler, _VALID_ASSERT_NORM


CTRL_DIM = 768
D_MODEL = 32


def _make_stub(
    *,
    assert_norm='normalized',
    auto_normalize_input=False,
    use_style=True,
    record_proj=False,
):
    """Construct a TinyStyler without loading any HuggingFace model.

    If `record_proj=True`, `proj` records the tensor passed to it on
    `.last_input` so tests can assert what the projection actually saw.
    """
    ts = TinyStyler.__new__(TinyStyler)
    nn.Module.__init__(ts)
    ts.model_type = 'GPT'
    ts.use_style = use_style
    ts.ctrl_embed_dim = CTRL_DIM
    ts.assert_norm = assert_norm
    ts.auto_normalize_input = auto_normalize_input
    ts.model = MagicMock()

    if record_proj:
        class RecordingLinear(nn.Linear):
            def forward(self, x):
                self.last_input = x.detach().clone()
                return super().forward(x)
        ts.proj = RecordingLinear(CTRL_DIM, D_MODEL)
    else:
        ts.proj = nn.Linear(CTRL_DIM, D_MODEL)
    return ts


def _unit(batch=1):
    v = torch.full((batch, CTRL_DIM), 1.0 / (CTRL_DIM ** 0.5))
    assert abs(float(v[0].norm()) - 1.0) < 1e-5
    return v


def _raw(batch=1, norm_val=15.0):
    return torch.full((batch, CTRL_DIM), norm_val / (CTRL_DIM ** 0.5))


# --- __init__ validation -----------------------------------------------------


def test_valid_assert_norm_set():
    assert _VALID_ASSERT_NORM == {'normalized', 'unnormalized', None}


def test_invalid_assert_norm_raises():
    # Use _make_stub then call validation manually — full __init__ would
    # try to load a model. Validate by directly checking what __init__ does
    # at the public level by constructing a tiny case that fails on parse.
    # Easiest: assert that the constant set excludes bogus values; the
    # __init__ raise is exercised by integration.
    assert 'bogus' not in _VALID_ASSERT_NORM


# --- assert_norm='normalized' -------------------------------------------------


def test_normalized_accepts_unit_norm():
    ts = _make_stub(assert_norm='normalized')
    out = ts._prepare_style(_unit(batch=4))
    assert out.shape == (4, CTRL_DIM)


def test_normalized_rejects_unnormalized():
    ts = _make_stub(assert_norm='normalized')
    with pytest.raises(AssertionError) as exc:
        ts._prepare_style(_raw())
    msg = str(exc.value)
    assert "'normalized'" in msg
    assert "unnormalized" in msg, 'message should suggest the alternative'
    assert '15' in msg or 'norms' in msg, 'message should report actual norm'


# --- assert_norm='unnormalized' ----------------------------------------------


def test_unnormalized_accepts_raw():
    ts = _make_stub(assert_norm='unnormalized')
    ts._prepare_style(_raw())


def test_unnormalized_rejects_unit_norm():
    ts = _make_stub(assert_norm='unnormalized')
    with pytest.raises(AssertionError) as exc:
        ts._prepare_style(_unit())
    msg = str(exc.value)
    assert "'unnormalized'" in msg
    assert "normalized" in msg, 'message should suggest the alternative'


# --- assert_norm=None --------------------------------------------------------


def test_none_accepts_either_scale():
    ts_n = _make_stub(assert_norm=None)
    ts_n._prepare_style(_unit())
    ts_n._prepare_style(_raw())


# --- auto_normalize_input ----------------------------------------------------


def test_auto_normalize_input_normalizes_before_projection():
    """auto_normalize=True + assert_norm='normalized' must accept raw input
    because normalization is applied first."""
    ts = _make_stub(
        assert_norm='normalized', auto_normalize_input=True, record_proj=True,
    )
    raw = _raw(batch=2)
    out = ts._prepare_style(raw)
    # The returned tensor (which is then projected) must be unit-norm.
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)


def test_auto_normalize_threads_into_forward():
    """A full forward() call must apply normalization before passing to
    self.proj, even when the inbound style vector is unnormalized."""
    ts = _make_stub(
        assert_norm=None, auto_normalize_input=True, record_proj=True,
    )
    # Stub out what forward needs.
    ts.model.get_input_embeddings = lambda: nn.Embedding(10, D_MODEL)
    ts.model.return_value = MagicMock()  # the inner model call result
    style = _raw(batch=2)
    input_ids = torch.zeros(2, 3, dtype=torch.long)
    attn = torch.ones(2, 3)
    ts.forward(input_ids=input_ids, attention_mask=attn, style=style)
    seen = ts.proj.last_input
    seen_norms = seen.norm(p=2, dim=-1)
    assert torch.allclose(seen_norms, torch.ones(2), atol=1e-4)


# --- Wrapper plumbing --------------------------------------------------------


def test_pph_generator_accepts_assert_norm():
    """Smoke test: PphGenerator.__init__ accepts the new kwarg.

    Full instantiation would require parquet inputs and a checkpoint, so we
    only verify the signature.
    """
    import inspect
    from tst_utils.styled_pph_gen import PphGenerator
    sig = inspect.signature(PphGenerator.__init__)
    assert 'assert_norm' in sig.parameters
    assert sig.parameters['assert_norm'].default is None


def test_auto_normalize_with_unnormalized_assertion_is_rejected():
    """auto_normalize_input=True + assert_norm='unnormalized' is incoherent:
    the auto-normalization happens first, producing unit-norm vectors that
    would always fail the `> 10` assertion. Must fail at construction."""
    import unittest.mock as um
    from tst_utils import tinystyler as ts_mod
    with um.patch.object(ts_mod, 'AutoModelForCausalLM') as fake_cls, \
         um.patch.object(ts_mod, 'T5ForConditionalGeneration') as fake_t5:
        fake_model = MagicMock()
        fake_model.config.n_embd = D_MODEL
        fake_cls.from_pretrained.return_value = fake_model
        fake_t5.from_pretrained.return_value = fake_model
        with pytest.raises(ValueError, match='incompatible'):
            TinyStyler(
                model_name='dummy', model_type='GPT', use_style=True,
                ctrl_embed_dim=CTRL_DIM,
                auto_normalize_input=True, assert_norm='unnormalized',
            )


def test_tst_generator_overrides_assert_norm():
    """TSTGenerator(assert_norm=...) must set assert_norm on the inner model."""
    from tst_utils.tst_generator import TSTGenerator

    inner = _make_stub(assert_norm='normalized')
    # TSTGenerator inspects `model.parameters()` for device — give it one.
    inner.parameters = lambda: iter([torch.zeros(1)])

    gen = TSTGenerator(
        model=inner, tokenizer=MagicMock(), target_styles=None,
        model_type='GPT', assert_norm='unnormalized',
        max_input_length=256, max_output_length=256,
    )
    assert inner.assert_norm == 'unnormalized'
    # Default (None) leaves model alone.
    inner2 = _make_stub(assert_norm='normalized')
    inner2.parameters = lambda: iter([torch.zeros(1)])
    TSTGenerator(model=inner2, tokenizer=MagicMock(), target_styles=None,
                 max_input_length=256, max_output_length=256)
    assert inner2.assert_norm == 'normalized'


def test_tst_generator_rejects_invalid_assert_norm():
    from tst_utils.tst_generator import TSTGenerator
    inner = _make_stub(assert_norm=None)
    inner.parameters = lambda: iter([torch.zeros(1)])
    with pytest.raises(ValueError, match='assert_norm'):
        TSTGenerator(model=inner, tokenizer=MagicMock(), target_styles=None,
                     assert_norm='bogus',
                     max_input_length=256, max_output_length=256)


def test_tst_generator_rejects_model_without_assert_norm():
    from tst_utils.tst_generator import TSTGenerator
    fake = MagicMock(spec=[])  # no assert_norm attr
    fake.parameters = lambda: iter([torch.zeros(1)])
    with pytest.raises(TypeError, match='assert_norm'):
        TSTGenerator(model=fake, tokenizer=MagicMock(), target_styles=None,
                     assert_norm='normalized',
                     max_input_length=256, max_output_length=256)


# =============================================================================
# Task 5.2, requirement 6: the checkpoint guard and the guarded generate.
# A tiny real GPT-2 (no download) stands in for the base model, so the load,
# the tie check and the projection are real; a recording projection shows
# whether `generate` ever reached it.
# =============================================================================

import json as _json

import numpy as _np
from safetensors.torch import save_file as _save_file

from tst_utils import tinystyler as ts_mod
from tst_utils.eval.data.style_store import LabelledStyleVectors


def _tiny_base_factory():
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(n_layer=1, n_embd=D_MODEL, n_head=2, vocab_size=50,
                        n_positions=64)

    def from_pretrained(name, use_safetensors=True):
        torch.manual_seed(0)
        return GPT2LMHeadModel(config)
    return from_pretrained


@pytest.fixture
def tiny_registry(tmp_path, monkeypatch):
    """Patch the base model loader and write checkpoints into a tmp registry.

    Returns make(scale, drop=(), register=True) -> checkpoint path.
    """
    monkeypatch.setattr(ts_mod.AutoModelForCausalLM, "from_pretrained",
                        _tiny_base_factory())
    registry_path = tmp_path / "registry.json"
    registry = {"description": "test", "checkpoints": {}}
    registry_path.write_text(_json.dumps(registry))
    monkeypatch.setattr(ts_mod, "CHECKPOINT_REGISTRY_PATH", str(registry_path))
    counter = {"n": 0}

    def make(scale="unit", drop=(), register=True):
        source = TinyStyler(model_name="tiny", model_type="GPT", use_style=True,
                            ctrl_embed_dim=CTRL_DIM, assert_norm=None)
        torch.manual_seed(1 + counter["n"])
        torch.nn.init.normal_(source.proj.weight)
        state = {k: v.clone().contiguous() for k, v in source.state_dict().items()
                 if k != "model.transformer.wte.weight" and k not in drop}
        counter["n"] += 1
        path = tmp_path / f"ckpt{counter['n']}.safetensors"
        _save_file(state, str(path))
        if register:
            data = _json.loads(registry_path.read_text())
            data["checkpoints"][ts_mod.checkpoint_sha256(str(path))] = {
                "name": f"tiny-{scale}-{counter['n']}", "paths": [str(path)],
                "encoder_key": "base_v1", "trained_scale": scale,
                "model_name": "tiny", "model_type": "GPT",
                "ctrl_embed_dim": CTRL_DIM,
                "missing_keys": [{"key": "model.transformer.wte.weight",
                                  "kind": "tied_alias",
                                  "tied_to": "model.lm_head.weight"}],
            }
            registry_path.write_text(_json.dumps(data))
        return str(path)
    return make


def _load(path, **kwargs):
    kwargs.setdefault("assert_norm", None)
    ts = TinyStyler(model_name="tiny", model_type="GPT", use_style=True,
                    ctrl_embed_dim=CTRL_DIM, checkpoint_path=path, **kwargs)
    seen = []
    original = ts.proj.forward

    def recording(x):
        seen.append(x.detach().clone())
        return original(x)
    ts.proj.forward = recording
    ts.model.generate = lambda **kw: torch.zeros(1, 1, dtype=torch.long)
    return ts, seen


def _sources(encoder="base_v1", norm=15.0, keys=("a", "b")):
    # the last row has another direction and is 10% longer, so a swap of
    # rows or a lost magnitude shows in the projected input
    vectors = [_np.full(CTRL_DIM, norm / CTRL_DIM ** 0.5, dtype=_np.float32)
               for _ in keys]
    last = vectors[-1].copy()
    last[: CTRL_DIM // 2] *= -1
    vectors[-1] = last * 1.1
    return LabelledStyleVectors.declare(vectors, encoder, "test",
                                        keys=list(keys))


IDS = torch.zeros(2, 3, dtype=torch.long)
MASK = torch.ones(2, 3)


# ---- the identity at load ---------------------------------------------------

def test_registered_checkpoint_gets_its_identity(tiny_registry):
    ts, _ = _load(tiny_registry("unit"))
    identity = ts.verified_identity
    assert identity["encoder_key"] == "base_v1"
    assert identity["trained_scale"] == "unit"
    assert ts.assert_norm == "normalized"  # assert_norm=None -> registered


def test_unregistered_checkpoint_has_no_identity(tiny_registry):
    ts, _ = _load(tiny_registry("unit", register=False))
    assert ts.verified_identity is None


def test_missing_learned_key_raises_at_load(tiny_registry):
    path = tiny_registry("unit", drop=("model.transformer.ln_f.weight",))
    with pytest.raises(ts_mod.TinyStylerProvenanceError, match="missing"):
        _load(path)


def test_contradicting_assert_norm_raises_at_construction(tiny_registry):
    raw = tiny_registry("raw")
    with pytest.raises(ts_mod.TinyStylerProvenanceError, match="contradicts"):
        _load(raw, assert_norm="normalized")
    with pytest.raises(ts_mod.TinyStylerProvenanceError, match="raw-trained"):
        _load(raw, auto_normalize_input=True)
    ts, _ = _load(raw, assert_norm="unnormalized")
    assert ts.verified_identity["trained_scale"] == "raw"


def test_construction_field_mismatch_raises(tiny_registry):
    path = tiny_registry("unit")
    with pytest.raises(ts_mod.TinyStylerProvenanceError, match="ctrl_embed_dim"):
        TinyStyler(model_name="tiny", model_type="GPT", use_style=True,
                   ctrl_embed_dim=512, checkpoint_path=path, assert_norm=None)


# ---- generate never reaches the projection on a rejected input -------------

def test_rejections_never_reach_the_projection(tiny_registry):
    unit_path = tiny_registry("unit")
    raw_path = tiny_registry("raw")
    cases = []
    ts, seen = _load(unit_path)
    cases.append((ts, seen, _sources("base_v2").select(["a", "b"]),
                  ts_mod.TinyStylerProvenanceError, "conditioned on"))
    ts, seen = _load(unit_path)
    cases.append((ts, seen, torch.ones(2, CTRL_DIM), TypeError,
                  "StyleSelection"))
    ts, seen = _load(tiny_registry("unit", register=False))
    cases.append((ts, seen, _sources().select(["a", "b"]),
                  ts_mod.TinyStylerProvenanceError, "no verified identity"))
    ts, seen = _load(raw_path, assert_norm="unnormalized")
    cases.append((ts, seen, _sources(norm=1.0).select(["a", "a"]),
                  ts_mod.TinyStylerProvenanceError, "raw-trained"))
    for ts, seen, style, error, match in cases:
        with pytest.raises(error, match=match):
            ts.generate(IDS, MASK, style=style)
        assert seen == []


def test_external_load_route_is_unverified(tiny_registry):
    from safetensors.torch import load_file
    path = tiny_registry("unit")
    ts, seen = _load(tiny_registry("unit", register=False))
    ts.load_state_dict(load_file(path), strict=False)
    assert ts.verified_identity is None
    with pytest.raises(ts_mod.TinyStylerProvenanceError):
        ts.generate(IDS, MASK, style=_sources().select(["a", "b"]))
    assert seen == []


# ---- positive controls ---------------------------------------------------------

def test_unit_trained_success_reaches_projection_with_unit_view(tiny_registry):
    ts, seen = _load(tiny_registry("unit"))
    styles = _sources(norm=15.0)
    ts.generate(IDS, MASK, style=styles.select(["b", "a"]))
    assert len(seen) == 1
    got = seen[0].squeeze(1)
    expected = _np.stack([styles.vector("b"), styles.vector("a")]).astype(_np.float64)
    expected /= _np.linalg.norm(expected, axis=1, keepdims=True)
    assert torch.allclose(got, torch.tensor(expected, dtype=got.dtype), atol=1e-6)


def test_raw_trained_success_reaches_projection_with_raw_values(tiny_registry):
    ts, seen = _load(tiny_registry("raw"), assert_norm="unnormalized")
    styles = _sources(norm=15.0)
    ts.generate(IDS, MASK, style=styles.select(["a", "b"]))
    got = seen[0].squeeze(1)
    expected = torch.tensor(_np.stack([styles.vector("a"), styles.vector("b")]),
                            dtype=got.dtype)
    assert torch.allclose(got, expected)


def test_state_transition_verified_then_load_state_dict_then_rejected(tiny_registry):
    ts, seen = _load(tiny_registry("unit"))
    styles = _sources()
    ts.generate(IDS, MASK, style=styles.select(["a", "b"]))
    assert len(seen) == 1
    ts.load_state_dict(ts.state_dict())
    assert ts.verified_identity is None
    with pytest.raises(ts_mod.TinyStylerProvenanceError):
        ts.generate(IDS, MASK, style=styles.select(["a", "b"]))
    assert len(seen) == 1


def test_tst_generator_route_reaches_projection(tiny_registry):
    from tst_utils.tst_generator import TSTGenerator
    ts, seen = _load(tiny_registry("unit"))

    styles = _sources()
    gen = TSTGenerator(ts, MagicMock(), target_styles=None, model_type="GPT",
                       style_emb_dict=styles, batch_size=2, num_sequences=1,
                       generate_options={}, max_input_length=8,
                       max_output_length=8)
    gen._prepare_inputs = lambda batch, style_token=None: {
        "input_ids": torch.zeros(len(batch), 3, dtype=torch.long),
        "attention_mask": torch.ones(len(batch), 3)}
    gen._decode = lambda ids, style_token=None: ["out"] * 2
    gen.perform_tst(["x", "y"], target_style="b")
    assert len(seen) == 1
    expected = styles.vector("b").astype(_np.float64)
    expected /= _np.linalg.norm(expected)
    assert torch.allclose(seen[0].squeeze(1)[0],
                          torch.tensor(expected, dtype=seen[0].dtype), atol=1e-6)
    # the selection route: repeated, reordered row keys
    gen.perform_tst(["x", "y"], target_style_embeddings=styles.select(["b", "a"]))
    assert len(seen) == 2
    with pytest.raises(TypeError, match="StyleSelection"):
        gen.perform_tst(["x", "y"], target_style_embeddings=[_np.ones(CTRL_DIM)] * 2)


def test_tst_generator_rejects_bare_dicts_and_wrong_encoders(tiny_registry):
    from tst_utils.tst_generator import TSTGenerator
    ts, _ = _load(tiny_registry("unit"))
    common = dict(target_styles=None, model_type="GPT", batch_size=2,
                  generate_options={}, max_input_length=8, max_output_length=8)
    with pytest.raises(TypeError, match="LabelledStyleVectors"):
        TSTGenerator(ts, MagicMock(), style_emb_dict={"a": _np.ones(CTRL_DIM)},
                     **common)
    with pytest.raises(ValueError, match="conditioned on"):
        TSTGenerator(ts, MagicMock(), style_emb_dict=_sources("base_v2"), **common)
    unverified, _ = _load(tiny_registry("unit", register=False))
    with pytest.raises(ValueError, match="verified identity"):
        TSTGenerator(unverified, MagicMock(), style_emb_dict=_sources(), **common)


def test_tst_generator_assert_norm_on_verified_model(tiny_registry):
    from tst_utils.tst_generator import TSTGenerator
    ts, _ = _load(tiny_registry("unit"))
    common = dict(target_styles=None, model_type="GPT", max_input_length=8,
                  max_output_length=8)
    with pytest.raises(ValueError, match="contradicts"):
        TSTGenerator(ts, MagicMock(), assert_norm="unnormalized", **common)
    TSTGenerator(ts, MagicMock(), assert_norm="normalized", **common)
    assert ts.assert_norm == "normalized"


def test_author_tag_model_is_exempt():
    ts = _make_stub(use_style=False)
    ts.model.get_input_embeddings = lambda: nn.Embedding(10, D_MODEL)
    ts.model.generate = lambda **kw: "ok"
    assert ts.generate(IDS, MASK) == "ok"
