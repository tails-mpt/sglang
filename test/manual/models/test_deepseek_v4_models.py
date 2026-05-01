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


def test_sparse_attn_v4_basic_shape():
    """sparse_attn_v4 produces the expected output shape and respects the
    -1 (invalid) entries in topk_idxs."""
    import torch
    from sglang.srt.models.deepseek_v4 import sparse_attn_v4

    bsz, seqlen, n_heads, head_dim = 2, 8, 4, 64
    kv_len, K_per_q = 16, 4

    torch.manual_seed(0)
    q = torch.randn(bsz, seqlen, n_heads, head_dim)
    kv = torch.randn(bsz, kv_len, head_dim)
    attn_sink = torch.randn(n_heads)
    # All valid indices in [0, kv_len)
    topk_idxs = torch.randint(0, kv_len, (bsz, seqlen, K_per_q)).int()
    softmax_scale = head_dim ** -0.5

    out = sparse_attn_v4(q, kv, attn_sink, topk_idxs, softmax_scale)
    assert out.shape == (bsz, seqlen, n_heads, head_dim)
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"


def test_sparse_attn_v4_invalid_indices_zero_contribution():
    """When topk_idxs contains -1 entries, those positions must contribute
    zero to the output (V4 reference behavior; matches the V4 kernel's
    masking)."""
    import torch
    from sglang.srt.models.deepseek_v4 import sparse_attn_v4

    bsz, seqlen, n_heads, head_dim = 1, 2, 2, 32
    kv_len, K_per_q = 8, 4

    torch.manual_seed(1)
    q = torch.randn(bsz, seqlen, n_heads, head_dim)
    kv = torch.randn(bsz, kv_len, head_dim)
    attn_sink = torch.zeros(n_heads)  # disable sink for cleaner check
    softmax_scale = head_dim ** -0.5

    # Case A: only valid index 0 per query (rest are -1)
    idxs_a = torch.full((bsz, seqlen, K_per_q), -1, dtype=torch.int32)
    idxs_a[..., 0] = 0
    # Case B: ALL -1 (no valid positions). With attn_sink=0, output should be
    # zeros (degenerate softmax mass goes entirely to the sink, which has v=0).
    idxs_b = torch.full((bsz, seqlen, K_per_q), -1, dtype=torch.int32)

    out_a = sparse_attn_v4(q, kv, attn_sink, idxs_a, softmax_scale)
    out_b = sparse_attn_v4(q, kv, attn_sink, idxs_b, softmax_scale)

    # When only kv[0] is valid, output should equal kv[0] for every query
    # (softmax weight is 1 on the single valid position; sink=0 contributes
    # to denominator but the only-valid softmax weight stays > 0).
    assert torch.allclose(
        out_a[0, 0, 0], kv[0, 0], atol=1e-5
    ), "single-valid-idx output should equal that kv"

    # When NO indices valid, the output is dominated by the sink and the
    # values are zero except for floating-point noise.
    assert torch.allclose(out_b, torch.zeros_like(out_b), atol=1e-5), (
        "all-invalid-idx output should be zero (sink absorbs all mass)"
    )


@pytest.mark.skip(reason="Requires DeepseekV4ForCausalLM full trunk + load_weights")
def test_v4attention_forward_shape():
    """V4Attention forward shape test — needs full config + initialized
    parameters; turned on once load_weights ports a real checkpoint."""
    pass


@pytest.mark.skip(reason="Requires DeepseekV4ForCausalLM full trunk + sparse_attn_v4")
def test_eagle3_aux_capture_shape():
    """End-to-end aux capture: forward returns (logits, list of [B,T,d] aux tensors)."""
    pass


@pytest.mark.skip(reason="Requires load_weights body (TODO(phase1-loader))")
def test_load_weights_from_hf_checkpoint():
    """Load V4-Flash weights from HF checkpoint into the model."""
    pass


# =====================================================================
# Key-remap test stub (for the GPU resume agent picking up load_weights)
# =====================================================================
# These tests don't need a real V4 checkpoint. They drive a future
# `_remap_v4_checkpoint_keys` helper (to be added as a module-level
# function in deepseek_v4.py) that translates HF checkpoint key names
# to our parameter names. The mismatches are documented in detail in
# DeepseekV4ForCausalLM.load_weights's docstring (commit 5e4dba418a).
#
# To enable these tests:
#   1. Add a helper `_remap_v4_checkpoint_keys(state_dict: dict, config
#      ) -> dict` to deepseek_v4.py. The helper should:
#      - Rename top-level: head.weight -> lm_head.weight
#      - Rename per-layer norms: <prefix>.norm.weight -> <prefix>.norm_weight
#        (q_norm, kv_norm, attn_norm, ffn_norm, compressor.norm)
#      - Move mHC params: layers.<i>.hc_<x>_<y> -> layers.<i>.hc.hc_<x>_<y>
#        (where <x> in {attn, ffn} and <y> in {fn, base, scale})
#   2. Remove the `pytest.mark.skip` decorator from each test below.
# These tests do NOT cover FP4 dequant — that's separately validated
# against a real V4 checkpoint on a GPU (see gpu-handoff.md "Step 1").


def _make_synthetic_v4_checkpoint_subset(num_layers=4, n_routed_experts=8):
    """Build a small synthetic state_dict mimicking V4 checkpoint key naming.

    Subset only — covers the keys the remap helper needs to rewrite,
    not the full 69187-key checkpoint. One sample of each pattern from
    DeepseekV4ForCausalLM.load_weights's docstring.
    """
    import torch
    sd = {}

    # Top-level
    sd["embed.weight"] = torch.zeros(256, 64)
    sd["head.weight"] = torch.zeros(256, 64)  # ← rename to lm_head.weight
    sd["hc_head_fn"] = torch.zeros(4, 4 * 64)
    sd["hc_head_base"] = torch.zeros(4)
    sd["hc_head_scale"] = torch.zeros(1)

    for i in range(num_layers):
        # Per-layer attn (sample of MQA + MLA Q/O paths)
        sd[f"layers.{i}.attn.attn_sink"] = torch.zeros(4)
        sd[f"layers.{i}.attn.q_norm.weight"] = torch.zeros(64)        # ← rename to q_norm_weight
        sd[f"layers.{i}.attn.kv_norm.weight"] = torch.zeros(64)       # ← rename to kv_norm_weight
        sd[f"layers.{i}.attn.wq_a.weight"] = torch.zeros(64, 64)
        sd[f"layers.{i}.attn.wq_b.weight"] = torch.zeros(64 * 4, 64)
        sd[f"layers.{i}.attn.wkv.weight"] = torch.zeros(64, 64)
        sd[f"layers.{i}.attn.wo_a.weight"] = torch.zeros(64, 64)
        sd[f"layers.{i}.attn.wo_b.weight"] = torch.zeros(64, 64)

        # Per-layer norms (RMSNorm submodule .weight in checkpoint)
        sd[f"layers.{i}.attn_norm.weight"] = torch.zeros(64)          # ← rename to attn_norm_weight
        sd[f"layers.{i}.ffn_norm.weight"] = torch.zeros(64)           # ← rename to ffn_norm_weight

        # mHC params (FLAT at layer level in checkpoint)
        sd[f"layers.{i}.hc_attn_fn"] = torch.zeros(24, 4 * 64)        # ← move under .hc
        sd[f"layers.{i}.hc_attn_base"] = torch.zeros(24)              # ← move under .hc
        sd[f"layers.{i}.hc_attn_scale"] = torch.zeros(3)              # ← move under .hc
        sd[f"layers.{i}.hc_ffn_fn"] = torch.zeros(24, 4 * 64)         # ← move under .hc
        sd[f"layers.{i}.hc_ffn_base"] = torch.zeros(24)               # ← move under .hc
        sd[f"layers.{i}.hc_ffn_scale"] = torch.zeros(3)               # ← move under .hc

        # MoE gate
        sd[f"layers.{i}.ffn.gate.weight"] = torch.zeros(n_routed_experts, 64)
        if i < 2:  # hash routing on first num_hash_layers=2 (in this synthetic)
            sd[f"layers.{i}.ffn.gate.tid2eid"] = torch.zeros(256, 2, dtype=torch.int32)
        else:
            sd[f"layers.{i}.ffn.gate.bias"] = torch.zeros(n_routed_experts)

        # Routed experts (one expert sampled)
        sd[f"layers.{i}.ffn.experts.0.w1.weight"] = torch.zeros(128, 64)
        sd[f"layers.{i}.ffn.experts.0.w2.weight"] = torch.zeros(64, 128)
        sd[f"layers.{i}.ffn.experts.0.w3.weight"] = torch.zeros(128, 64)

        # Shared expert
        sd[f"layers.{i}.ffn.shared_experts.w1.weight"] = torch.zeros(128, 64)
        sd[f"layers.{i}.ffn.shared_experts.w2.weight"] = torch.zeros(64, 128)
        sd[f"layers.{i}.ffn.shared_experts.w3.weight"] = torch.zeros(128, 64)

    return sd


def test_remap_top_level_keys():
    """Top-level renames: head.weight -> lm_head.weight (etc)."""
    from sglang.srt.models.deepseek_v4 import _remap_v4_checkpoint_keys  # noqa: F401

    sd_in = {"head.weight": object(), "embed.weight": object(), "hc_head_fn": object()}
    cfg = _make_full_v4_config()
    sd_out = _remap_v4_checkpoint_keys(sd_in, cfg)

    assert "head.weight" not in sd_out
    assert "lm_head.weight" in sd_out
    # These should pass through unchanged.
    assert "embed.weight" in sd_out
    assert "hc_head_fn" in sd_out


def test_remap_per_layer_norms():
    """Per-layer norms: <prefix>.norm.weight -> <prefix>.norm_weight."""
    from sglang.srt.models.deepseek_v4 import _remap_v4_checkpoint_keys

    sd_in = _make_synthetic_v4_checkpoint_subset(num_layers=2)
    cfg = _make_full_v4_config()
    sd_out = _remap_v4_checkpoint_keys(sd_in, cfg)

    for i in range(2):
        # Renames: q_norm, kv_norm, attn_norm, ffn_norm
        assert f"layers.{i}.attn.q_norm.weight" not in sd_out
        assert f"layers.{i}.attn.q_norm_weight" in sd_out
        assert f"layers.{i}.attn.kv_norm.weight" not in sd_out
        assert f"layers.{i}.attn.kv_norm_weight" in sd_out
        assert f"layers.{i}.attn_norm.weight" not in sd_out
        assert f"layers.{i}.attn_norm_weight" in sd_out
        assert f"layers.{i}.ffn_norm.weight" not in sd_out
        assert f"layers.{i}.ffn_norm_weight" in sd_out


def test_remap_mhc_to_submodule():
    """mHC params move from flat layer level to V4HCBlock submodule."""
    from sglang.srt.models.deepseek_v4 import _remap_v4_checkpoint_keys

    sd_in = _make_synthetic_v4_checkpoint_subset(num_layers=3)
    cfg = _make_full_v4_config()
    sd_out = _remap_v4_checkpoint_keys(sd_in, cfg)

    for i in range(3):
        for x in ("attn", "ffn"):
            for y in ("fn", "base", "scale"):
                old = f"layers.{i}.hc_{x}_{y}"
                new = f"layers.{i}.hc.hc_{x}_{y}"
                assert old not in sd_out, f"old key {old} still present"
                assert new in sd_out, f"new key {new} missing"


# Full set of 82 distinct V4-Flash checkpoint key patterns extracted on
# 2026-04-30 from /tmp/v4-flash-meta/model.safetensors.index.json. One
# representative key per pattern (digits substituted to "0"). Fixture for
# the comprehensive remap test below.
_V4_KEY_PATTERNS = (
    "embed.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "head.weight",
    "norm.weight",
    "layers.0.attn.attn_sink",
    "layers.0.attn.compressor.ape",
    "layers.0.attn.compressor.norm.weight",
    "layers.0.attn.compressor.wgate.weight",
    "layers.0.attn.compressor.wkv.weight",
    "layers.0.attn.indexer.compressor.ape",
    "layers.0.attn.indexer.compressor.norm.weight",
    "layers.0.attn.indexer.compressor.wgate.weight",
    "layers.0.attn.indexer.compressor.wkv.weight",
    "layers.0.attn.indexer.weights_proj.weight",
    "layers.0.attn.indexer.wq_b.scale",
    "layers.0.attn.indexer.wq_b.weight",
    "layers.0.attn.kv_norm.weight",
    "layers.0.attn.q_norm.weight",
    "layers.0.attn.wkv.scale",
    "layers.0.attn.wkv.weight",
    "layers.0.attn.wo_a.scale",
    "layers.0.attn.wo_a.weight",
    "layers.0.attn.wo_b.scale",
    "layers.0.attn.wo_b.weight",
    "layers.0.attn.wq_a.scale",
    "layers.0.attn.wq_a.weight",
    "layers.0.attn.wq_b.scale",
    "layers.0.attn.wq_b.weight",
    "layers.0.attn_norm.weight",
    "layers.0.ffn.experts.0.w0.scale",
    "layers.0.ffn.experts.0.w0.weight",
    "layers.0.ffn.gate.bias",
    "layers.0.ffn.gate.tid2eid",
    "layers.0.ffn.gate.weight",
    "layers.0.ffn.shared_experts.w0.scale",
    "layers.0.ffn.shared_experts.w0.weight",
    "layers.0.ffn_norm.weight",
    "layers.0.hc_attn_base",
    "layers.0.hc_attn_fn",
    "layers.0.hc_attn_scale",
    "layers.0.hc_ffn_base",
    "layers.0.hc_ffn_fn",
    "layers.0.hc_ffn_scale",
    "mtp.0.attn.attn_sink",
    "mtp.0.attn.kv_norm.weight",
    "mtp.0.attn.q_norm.weight",
    "mtp.0.attn.wkv.scale",
    "mtp.0.attn.wkv.weight",
    "mtp.0.attn.wo_a.scale",
    "mtp.0.attn.wo_a.weight",
    "mtp.0.attn.wo_b.scale",
    "mtp.0.attn.wo_b.weight",
    "mtp.0.attn.wq_a.scale",
    "mtp.0.attn.wq_a.weight",
    "mtp.0.attn.wq_b.scale",
    "mtp.0.attn.wq_b.weight",
    "mtp.0.attn_norm.weight",
    "mtp.0.e_proj.scale",
    "mtp.0.e_proj.weight",
    "mtp.0.enorm.weight",
    "mtp.0.ffn.experts.0.w0.scale",
    "mtp.0.ffn.experts.0.w0.weight",
    "mtp.0.ffn.gate.bias",
    "mtp.0.ffn.gate.weight",
    "mtp.0.ffn.shared_experts.w0.scale",
    "mtp.0.ffn.shared_experts.w0.weight",
    "mtp.0.ffn_norm.weight",
    "mtp.0.h_proj.scale",
    "mtp.0.h_proj.weight",
    "mtp.0.hc_attn_base",
    "mtp.0.hc_attn_fn",
    "mtp.0.hc_attn_scale",
    "mtp.0.hc_ffn_base",
    "mtp.0.hc_ffn_fn",
    "mtp.0.hc_ffn_scale",
    "mtp.0.hc_head_base",
    "mtp.0.hc_head_fn",
    "mtp.0.hc_head_scale",
    "mtp.0.hnorm.weight",
    "mtp.0.norm.weight",
    # Note: 82 patterns total; this list includes all 82 (1 per pattern).
)


def test_remap_all_82_v4_key_patterns():
    """Comprehensive test: remap each of V4-Flash's 82 distinct checkpoint
    key patterns and verify (a) total key count preserved, (b) every input
    key has a corresponding output key, (c) all `mtp.*` keys remain `mtp.*`
    so they can still be split out in Phase 2.

    Fixture is the actual key patterns extracted from
    /tmp/v4-flash-meta/model.safetensors.index.json on 2026-04-30. If
    DeepSeek changes the V4 checkpoint key layout in a future release, this
    test breaks loudly; that's the right behavior.
    """
    import torch
    from sglang.srt.models.deepseek_v4 import _remap_v4_checkpoint_keys

    # Build a synthetic state_dict where every key is a sentinel tensor.
    # The test only cares about KEYS, not VALUES.
    sd_in = {k: torch.zeros(1) for k in _V4_KEY_PATTERNS}
    cfg = _make_full_v4_config()

    sd_out = _remap_v4_checkpoint_keys(sd_in, cfg)

    # (a) Total key count preserved (no dropped or doubled keys)
    assert len(sd_out) == len(sd_in) == len(_V4_KEY_PATTERNS), (
        f"Key count changed: {len(sd_in)} -> {len(sd_out)}; this means a "
        f"key was dropped or duplicated by the remap."
    )

    # (b) Every input pattern produces a non-None output key (string).
    # We can't assert specific renames for all 82 (many pass through), but
    # we can verify no `None` or empty keys.
    for k_out in sd_out:
        assert isinstance(k_out, str) and len(k_out) > 0

    # (c) All input mtp.* keys remain mtp.* in output, so the
    # _split_v4_mtp_keys downstream filter still works.
    for k_in in _V4_KEY_PATTERNS:
        if k_in.startswith("mtp."):
            # Find the matching output key. Since the remap may have renamed
            # it (e.g., mtp.0.attn.q_norm.weight -> mtp.0.attn.q_norm_weight),
            # we just verify SOME output key is mtp.*-prefixed corresponding
            # to this input.
            #
            # Due to 1-1 mapping, we can find it by elimination, but the
            # easier check: count how many keys are mtp.* in input and output.
            pass
    n_mtp_in = sum(1 for k in _V4_KEY_PATTERNS if k.startswith("mtp."))
    n_mtp_out = sum(1 for k in sd_out if k.startswith("mtp."))
    assert n_mtp_in == n_mtp_out, (
        f"mtp.* key count changed: input had {n_mtp_in}, output has "
        f"{n_mtp_out}. This breaks the _split_v4_mtp_keys assumption that "
        f"all MTP keys retain the mtp. prefix."
    )

    # (d) Spot-check: critical renames we verified separately above.
    assert "head.weight" not in sd_out and "lm_head.weight" in sd_out
    assert "norm.weight" not in sd_out and "final_norm_weight" in sd_out
    assert "layers.0.attn.q_norm.weight" not in sd_out
    assert "layers.0.attn.q_norm_weight" in sd_out
    assert "layers.0.hc_attn_fn" not in sd_out
    assert "layers.0.hc.hc_attn_fn" in sd_out
    assert "layers.0.attn.compressor.norm.weight" not in sd_out
    assert "layers.0.attn.compressor.norm_weight" in sd_out
    # Indexer compressor sub-norm uses the same pattern
    assert "layers.0.attn.indexer.compressor.norm.weight" not in sd_out
    assert "layers.0.attn.indexer.compressor.norm_weight" in sd_out


def test_remap_top_level_final_norm():
    """Top-level `norm.weight` (V4 trunk final RMSNorm) -> `final_norm_weight`.
    Discovered 2026-04-30 during V4 safetensors index audit; without this
    remap the trunk final norm tensor would be silently dropped at
    load_state_dict time."""
    import torch
    from sglang.srt.models.deepseek_v4 import _remap_v4_checkpoint_keys

    sd_in = {
        "norm.weight": torch.zeros(4096),
        "embed.weight": torch.zeros(129280, 4096),
        "head.weight": torch.zeros(129280, 4096),
    }
    cfg = _make_full_v4_config()
    sd_out = _remap_v4_checkpoint_keys(sd_in, cfg)
    assert "norm.weight" not in sd_out
    assert "final_norm_weight" in sd_out
    # head.weight still renames to lm_head.weight in the same call
    assert "lm_head.weight" in sd_out
    # embed.weight passes through
    assert "embed.weight" in sd_out


def test_remap_mtp_final_norm():
    """`mtp.<i>.norm.weight` (MTPBlock final RMSNorm) -> `mtp.<i>.norm_weight`.
    Same remap class as the per-layer norms but with the bare `norm` name."""
    import torch
    from sglang.srt.models.deepseek_v4 import _remap_v4_checkpoint_keys

    sd_in = {"mtp.0.norm.weight": torch.zeros(4096)}
    cfg = _make_full_v4_config()
    sd_out = _remap_v4_checkpoint_keys(sd_in, cfg)
    assert "mtp.0.norm.weight" not in sd_out
    assert "mtp.0.norm_weight" in sd_out


def test_split_mtp_keys():
    """_split_v4_mtp_keys partitions a state_dict into (main_dict, mtp_dict)
    based on the `mtp.` prefix per CLAUDE.md rule #12 (MTP loaded but not run).
    """
    import torch
    from sglang.srt.models.deepseek_v4 import _split_v4_mtp_keys

    sd_in = {
        "embed.weight": torch.zeros(256, 64),
        "lm_head.weight": torch.zeros(256, 64),
        "layers.0.attn.attn_sink": torch.zeros(4),
        "layers.0.ffn.gate.weight": torch.zeros(8, 64),
        "mtp.0.attn.wq_a.weight": torch.zeros(64, 64),
        "mtp.0.ffn_norm_weight": torch.zeros(64),
        "mtp.0.hc_attn_fn": torch.zeros(24, 256),
    }
    main, mtp = _split_v4_mtp_keys(sd_in)
    # 4 main + 3 mtp = 7 = total
    assert len(main) == 4
    assert len(mtp) == 3
    # Round-trip preservation
    assert set(main.keys()) | set(mtp.keys()) == set(sd_in.keys())
    assert set(main.keys()) & set(mtp.keys()) == set()
    # All mtp keys start with mtp.
    for k in mtp:
        assert k.startswith("mtp.")
    # No main keys start with mtp.
    for k in main:
        assert not k.startswith("mtp.")


def test_remap_pass_through_for_unrenamed_keys():
    """Keys that don't need remap pass through unchanged."""
    from sglang.srt.models.deepseek_v4 import _remap_v4_checkpoint_keys

    sd_in = _make_synthetic_v4_checkpoint_subset(num_layers=2)
    cfg = _make_full_v4_config()
    n_in = len(sd_in)
    sd_out = _remap_v4_checkpoint_keys(sd_in, cfg)

    # Total key count preserved (every input key produces exactly one output key).
    assert len(sd_out) == n_in

    # Spot-check: tid2eid, gate.weight, expert weights, attn_sink unchanged.
    for i in range(2):
        if i < 2:  # hash layers in our synthetic
            assert f"layers.{i}.ffn.gate.tid2eid" in sd_out
        assert f"layers.{i}.attn.attn_sink" in sd_out
        assert f"layers.{i}.ffn.gate.weight" in sd_out
        assert f"layers.{i}.ffn.experts.0.w1.weight" in sd_out
        assert f"layers.{i}.ffn.shared_experts.w2.weight" in sd_out
        assert f"layers.{i}.attn.wq_a.weight" in sd_out
