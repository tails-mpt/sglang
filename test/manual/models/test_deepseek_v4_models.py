# Copyright 2026 ThoughtWorks (Crucible). Licensed under Apache-2.0.
# ==============================================================================
# Scaffolding-level interface contract for the new DeepseekV4ForCausalLM
# model class. Validates:
#   - The module imports without error
#   - DeepseekV4ForCausalLM and EntryClass are exported
#   - set_eagle3_layers_to_capture stores the configured layer indices
#   - enable_return_hidden_states getter/setter works
#   - V4LayerAttentionType.from_compress_ratio dispatches correctly for
#     compress_ratio values 0, 4, 128 and raises on others
#
# Does NOT exercise any forward pass — every component is currently a
# scaffold that raises NotImplementedError. Forward-pass tests land in
# subsequent commits as each component is ported.
#
# Run:
#   cd sglang && pytest test/manual/models/test_deepseek_v4_models.py -v
# (or the equivalent on a CI runner that has sglang's deps installed)
# ==============================================================================

import pytest


def test_module_imports():
    """The deepseek_v4 module imports without raising."""
    from sglang.srt.models import deepseek_v4  # noqa: F401


def test_entry_class_is_exported():
    """sglang's registry uses pkgutil to walk models/; the class name must
    match HF config.architectures[0] = "DeepseekV4ForCausalLM"."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, EntryClass

    assert EntryClass is DeepseekV4ForCausalLM
    assert DeepseekV4ForCausalLM.__name__ == "DeepseekV4ForCausalLM"


def test_layer_attention_type_dispatch():
    """Per-layer attention dispatch is the V4-specific switch on
    config.compress_ratios[layer_id]."""
    from sglang.srt.models.deepseek_v4 import V4LayerAttentionType

    assert V4LayerAttentionType.from_compress_ratio(0) == V4LayerAttentionType.WINDOW_ONLY
    assert V4LayerAttentionType.from_compress_ratio(4) == V4LayerAttentionType.CSA
    assert V4LayerAttentionType.from_compress_ratio(128) == V4LayerAttentionType.HCA

    # Unsupported ratios must raise — V4-Flash's compress_ratios array only
    # contains {0, 4, 128}; anything else means the V4 release has introduced
    # a new layer type and we need to extend the enum.
    with pytest.raises(ValueError, match="unsupported compress_ratio"):
        V4LayerAttentionType.from_compress_ratio(2)
    with pytest.raises(ValueError, match="unsupported compress_ratio"):
        V4LayerAttentionType.from_compress_ratio(64)


def _make_minimal_v4_config():
    """Return a minimal HF-style config object that satisfies
    DeepseekV4ForCausalLM.__init__ (without triggering any of the
    NotImplementedError-raising sub-component constructors)."""
    from transformers import PretrainedConfig

    cfg = PretrainedConfig()
    cfg.hidden_size = 4096
    cfg.num_hidden_layers = 43
    cfg.vocab_size = 129280
    cfg.compress_ratios = [
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 0,
    ]
    cfg.hc_mult = 4
    cfg.hc_sinkhorn_iters = 20
    cfg.hc_eps = 1e-6
    cfg.num_hash_layers = 3
    cfg.rms_norm_eps = 1e-6
    return cfg


def test_eagle3_layers_to_capture_storage():
    """The Eagle3 hook stores configured layer indices on the model object."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    cfg = _make_minimal_v4_config()
    model = DeepseekV4ForCausalLM(cfg)

    # Default: no layers configured.
    assert model._eagle3_layers_to_capture == []

    # Set the proposed V4 slot triple [1, 21, 41].
    model.set_eagle3_layers_to_capture([1, 21, 41])
    assert model._eagle3_layers_to_capture == [1, 21, 41]

    # Re-setting overwrites; lists are stored by value (new list, not aliased).
    new_layers = [2, 22, 42]
    model.set_eagle3_layers_to_capture(new_layers)
    assert model._eagle3_layers_to_capture == [2, 22, 42]
    new_layers.append(43)  # mutate original; should not affect stored list
    assert model._eagle3_layers_to_capture == [2, 22, 42]


def test_enable_return_hidden_states_property():
    """The getter/setter for the aux-hidden-state-return flag works."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    cfg = _make_minimal_v4_config()
    model = DeepseekV4ForCausalLM(cfg)

    assert model.enable_return_hidden_states is False
    model.enable_return_hidden_states = True
    assert model.enable_return_hidden_states is True
    model.enable_return_hidden_states = False
    assert model.enable_return_hidden_states is False
    # Truthy non-bool -> bool coerced
    model.enable_return_hidden_states = 1
    assert model.enable_return_hidden_states is True


@pytest.mark.skip(reason="Phase 1 port pending — every sub-component currently raises NotImplementedError")
def test_forward_shape():
    """End-to-end forward shape test — turned on as components land."""
    pass


@pytest.mark.skip(reason="Phase 1 port pending — load_weights body not yet written")
def test_load_weights_fp4_path():
    """FP4 expert weight loading test — turned on when the weight loader is ported."""
    pass
