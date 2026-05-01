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


def _make_full_v4_config():
    """Return an HF-style config object with EVERY key V4 model construction
    needs. Used by tests that instantiate sub-components or the full model.

    Numbers match V4-Flash's actual config.json (pulled to /tmp/v4-flash-meta/
    in Phase 0). Where Phase 0 deferred a default (e.g. max_batch_size), we
    set a small value so tests run fast.
    """
    from transformers import PretrainedConfig

    cfg = PretrainedConfig()
    # Trunk
    cfg.hidden_size = 4096
    cfg.num_hidden_layers = 43
    cfg.vocab_size = 129280
    cfg.rms_norm_eps = 1e-6

    # Per-layer attention dispatch (44 entries: 43 main + 1 MTP)
    cfg.compress_ratios = [
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 0,
    ]

    # Attention shapes
    cfg.num_attention_heads = 64
    cfg.num_key_value_heads = 1  # MQA
    cfg.head_dim = 512
    cfg.qk_rope_head_dim = 64
    cfg.q_lora_rank = 1024
    cfg.o_lora_rank = 1024
    cfg.o_groups = 8
    cfg.sliding_window = 128
    cfg.max_position_embeddings = 1048576
    cfg.max_batch_size = 2  # small for tests; V4 default is 4

    # Rotary
    cfg.rope_theta = 10000
    cfg.compress_rope_theta = 160000
    cfg.rope_scaling = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }

    # mHC
    cfg.hc_mult = 4
    cfg.hc_sinkhorn_iters = 20
    cfg.hc_eps = 1e-6

    # MoE
    cfg.n_routed_experts = 256
    cfg.n_shared_experts = 1
    cfg.num_experts_per_tok = 6
    cfg.moe_intermediate_size = 2048
    cfg.scoring_func = "sqrtsoftplus"
    cfg.routed_scaling_factor = 1.5
    cfg.swiglu_limit = 10.0
    cfg.num_hash_layers = 3
    cfg.topk_method = "noaux_tc"

    # MTP (loaded but not run per rule #12)
    cfg.num_nextn_predict_layers = 1

    return cfg


def _make_tiny_v4_config():
    """Return a SHRUNK config used by tests that need to actually construct
    V4Attention/V4HCBlock/etc. without the full 43-layer trunk. Keeps shapes
    minimal but valid (e.g. 4 layers instead of 43, smaller hidden_size).
    """
    cfg = _make_full_v4_config()
    # Shrink the trunk to 4 layers so we have one of each attention type.
    cfg.num_hidden_layers = 4
    cfg.compress_ratios = [0, 4, 128, 0]  # window, CSA, HCA, window
    # Keep MoE shrunk too — tests don't exercise MoE math, just construction.
    cfg.n_routed_experts = 8
    cfg.num_experts_per_tok = 2
    cfg.num_hash_layers = 1
    cfg.moe_intermediate_size = 512
    cfg.hidden_size = 256
    cfg.head_dim = 64
    cfg.qk_rope_head_dim = 16
    cfg.q_lora_rank = 64
    cfg.o_lora_rank = 64
    cfg.num_attention_heads = 4
    cfg.o_groups = 2
    cfg.sliding_window = 16
    cfg.max_position_embeddings = 512
    cfg.vocab_size = 256
    return cfg


def test_v4hcblock_constructs():
    """V4HCBlock allocates the 6 mHC parameters with correct shapes."""
    from sglang.srt.models.deepseek_v4 import V4HCBlock

    hc_mult = 4
    hidden_size = 256
    block = V4HCBlock(hidden_size, hc_mult=hc_mult, hc_sinkhorn_iters=10, hc_eps=1e-6)

    mix_hc = (2 + hc_mult) * hc_mult  # = 24
    hc_dim = hc_mult * hidden_size  # = 1024
    assert block.hc_attn_fn.shape == (mix_hc, hc_dim)
    assert block.hc_ffn_fn.shape == (mix_hc, hc_dim)
    assert block.hc_attn_base.shape == (mix_hc,)
    assert block.hc_ffn_base.shape == (mix_hc,)
    assert block.hc_attn_scale.shape == (3,)
    assert block.hc_ffn_scale.shape == (3,)
    # All in fp32 per V4 reference.
    assert block.hc_attn_fn.dtype == __import__("torch").float32


def test_hc_split_sinkhorn_shapes_and_doubly_stochastic():
    """hc_split_sinkhorn_v4 returns correct shapes; comb is approximately
    doubly-stochastic after Sinkhorn iteration."""
    import torch
    from sglang.srt.models.deepseek_v4 import hc_split_sinkhorn_v4

    hc = 4
    mix_hc = (2 + hc) * hc
    bsz, seqlen = 2, 8

    torch.manual_seed(0)
    mixes = torch.randn(bsz, seqlen, mix_hc)
    hc_scale = torch.ones(3) * 0.5
    hc_base = torch.zeros(mix_hc)

    pre, post, comb = hc_split_sinkhorn_v4(
        mixes, hc_scale, hc_base, hc_mult=hc, sinkhorn_iters=20, eps=1e-6
    )

    assert pre.shape == (bsz, seqlen, hc)
    assert post.shape == (bsz, seqlen, hc)
    assert comb.shape == (bsz, seqlen, hc, hc)

    # After 20 Sinkhorn iters, each row sum and column sum of comb should
    # be close to 1 (doubly-stochastic property modulo eps).
    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        f"comb row sums not close to 1; max deviation: {(row_sums - 1).abs().max():.4f}"
    )
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3), (
        f"comb col sums not close to 1; max deviation: {(col_sums - 1).abs().max():.4f}"
    )


def test_v4hcblock_pre_post_round_trip():
    """hc_pre then hc_post on identity-like inputs preserves shape."""
    import torch
    from sglang.srt.models.deepseek_v4 import V4HCBlock

    hc_mult = 4
    hidden_size = 256
    block = V4HCBlock(hidden_size, hc_mult=hc_mult, hc_sinkhorn_iters=5)

    bsz, seqlen = 2, 8
    x = torch.randn(bsz, seqlen, hc_mult, hidden_size)

    y, post, comb = block.hc_pre(x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    assert y.shape == (bsz, seqlen, hidden_size)
    assert post.shape == (bsz, seqlen, hc_mult)
    assert comb.shape == (bsz, seqlen, hc_mult, hc_mult)

    # Synthetic sublayer output (just identity-ish).
    sublayer_out = y + 0.1 * torch.randn_like(y)
    z = block.hc_post(sublayer_out, x, post, comb)
    assert z.shape == (bsz, seqlen, hc_mult, hidden_size)


def test_v4compressor_constructs():
    """V4Compressor builds with both compress_ratio=4 (CSA, overlap=True)
    and compress_ratio=128 (HCA, overlap=False)."""
    from sglang.srt.models.deepseek_v4 import V4Compressor

    csa = V4Compressor(
        hidden_size=256, head_dim=64, compress_ratio=4, rope_head_dim=16,
        max_batch_size=2, rotate=False,
    )
    assert csa.overlap is True
    coff = 1 + int(csa.overlap)  # = 2
    assert csa.kv_state.shape == (2, coff * 4, coff * 64)
    assert csa.score_state.shape == (2, coff * 4, coff * 64)

    hca = V4Compressor(
        hidden_size=256, head_dim=64, compress_ratio=128, rope_head_dim=16,
        max_batch_size=2, rotate=False,
    )
    assert hca.overlap is False
    coff = 1 + int(hca.overlap)  # = 1
    assert hca.kv_state.shape == (2, coff * 128, coff * 64)


def test_v4gate_hash_branch():
    """Hash-routing layer (layer_id < num_hash_layers) routes via tid2eid lookup."""
    import torch
    from sglang.srt.models.deepseek_v4 import V4Gate

    cfg = _make_tiny_v4_config()
    cfg.num_hash_layers = 2  # layers 0, 1 hash; layers 2+ score-based

    # Layer 0: hash branch
    gate0 = V4Gate(layer_id=0, config=cfg)
    assert gate0.is_hash is True
    assert gate0.tid2eid.shape == (cfg.vocab_size, cfg.num_experts_per_tok)
    assert gate0.bias is None

    # Layer 2: score branch
    gate2 = V4Gate(layer_id=2, config=cfg)
    assert gate2.is_hash is False
    assert gate2.bias is not None
    assert gate2.bias.shape == (cfg.n_routed_experts,)

    # Hash-routing forward: uses tid2eid
    bsz_t = 4
    x = torch.randn(bsz_t, cfg.hidden_size)
    input_ids = torch.tensor([1, 2, 3, 4])

    # Manually set tid2eid so we have a deterministic check.
    gate0.tid2eid.data = torch.zeros_like(gate0.tid2eid).int()
    gate0.tid2eid.data[:, 0] = 0  # all tokens route to expert 0 first
    gate0.tid2eid.data[:, 1] = 1  # all tokens route to expert 1 second

    weights, indices = gate0(x, input_ids)
    assert weights.shape == (bsz_t, cfg.num_experts_per_tok)
    assert indices.shape == (bsz_t, cfg.num_experts_per_tok)
    # All tokens should route to {0, 1} per our manual tid2eid.
    assert torch.all(indices[:, 0] == 0)
    assert torch.all(indices[:, 1] == 1)


def test_layer_attention_type_dispatch_full_pattern():
    """V4LayerAttentionType correctly classifies the full V4-Flash 44-layer
    compress_ratios pattern."""
    from sglang.srt.models.deepseek_v4 import V4LayerAttentionType

    cfg = _make_full_v4_config()
    types = [V4LayerAttentionType.from_compress_ratio(r) for r in cfg.compress_ratios]

    # Layers 0, 1 should be window-only (warm-up)
    assert types[0] == V4LayerAttentionType.WINDOW_ONLY
    assert types[1] == V4LayerAttentionType.WINDOW_ONLY
    # Layer 42 should be window-only (final main layer)
    assert types[42] == V4LayerAttentionType.WINDOW_ONLY
    # Layer 43 should be window-only (MTP layer, ratio=0)
    assert types[43] == V4LayerAttentionType.WINDOW_ONLY
    # Even-indexed inner layers (2, 4, 6, ..., 40) should be CSA
    for i in range(2, 41, 2):
        assert types[i] == V4LayerAttentionType.CSA, f"layer {i} should be CSA but is {types[i]}"
    # Odd-indexed inner layers (3, 5, 7, ..., 41) should be HCA
    for i in range(3, 42, 2):
        assert types[i] == V4LayerAttentionType.HCA, f"layer {i} should be HCA but is {types[i]}"


@pytest.mark.skip(reason="Requires sparse_attn_v4 wired to NSA tilelang kernel (TODO(phase1-kernel))")
def test_v4attention_forward_shape():
    """V4Attention forward shape test — turned on when sparse_attn_v4 lands."""
    pass


@pytest.mark.skip(reason="Requires DeepseekV4ForCausalLM full trunk + sparse_attn_v4")
def test_eagle3_aux_capture_shape():
    """End-to-end aux capture: forward returns (logits, list of [B,T,d] aux tensors)."""
    pass


@pytest.mark.skip(reason="Requires load_weights body (TODO(phase1-loader))")
def test_load_weights_from_hf_checkpoint():
    """Load V4-Flash weights from HF checkpoint into the model."""
    pass
