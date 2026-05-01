# Copyright 2026 ThoughtWorks (Crucible). Licensed under Apache-2.0.
# ==============================================================================
# Inference-only DeepSeek-V4 model.
#
# DeepSeek-V4-Flash architecture (extracted from /tmp/v4-flash-meta/inference/model.py
# and the V4 technical report; full notes at
# experiments/DeepSeek-V4-Flash/architecture-notes.md):
#
#   - 43 main transformer blocks + 1 MTPBlock (`num_nextn_predict_layers=1`)
#   - hidden_size=4096, head_dim=512, n_heads=64, n_kv_heads=1 (MQA on K/V)
#   - q_lora_rank=1024 (low-rank Q proj wq_a -> q_norm -> wq_b)
#   - o_lora_rank=1024 with o_groups=8 (grouped low-rank O)
#   - 256 routed experts top-6 + 1 shared, sqrtsoftplus + noaux_tc routing
#   - First 3 MoE layers use HASH routing (`tid2eid` lookup, NOT score-based)
#   - FP8 (e4m3fn) non-expert weights with FP8 (e8m0fnu) scales, FP4
#     (e2m1fn_x2) expert weights
#   - Per-layer hybrid attention dispatch on `compress_ratios[layer_id]`:
#       0   = window-only (layers 0, 1, 42)
#       4   = CSA (Compressed Sparse Attention) — sliding window + Indexer-
#             driven sparse top-k from compressed KV (compress_ratio=4)
#       128 = HCA (Heavily Compressed Attention) — sliding window + fixed-
#             stride sparse selection from heavily compressed KV (no Indexer)
#   - mHC (Manifold-Constrained Hyper-Connections) hc_mult=4: maintains 4
#     copies of the hidden state, mixed via Sinkhorn-normalized weights.
#     Wraps both attn and ffn separately in each Block. NOT attention.
#   - YaRN scaling factor=16 from original_max_position_embeddings=65536 to
#     1M context. compress_rope_theta=160000 for compressed KV path; rope_theta
#     =10000 for sliding-window-only path.
#
# Reuse map (per architecture-notes.md "Reuse map for sglang fork"):
#   - V3.2 NSA infrastructure at sglang/srt/layers/attention/nsa/ covers
#     CSA's Indexer and sparse_attn kernels with compress_ratio=4. HCA reuses
#     the Compressor + sparse_attn but skips the Indexer.
#   - V3.2's deepseek_v2.py uses NSA via `is_deepseek_nsa(config)` gate at
#     line 1171 + `self.indexer = Indexer(...)` at line 1231 — same pattern
#     applies here.
#   - deepseek_common/ provides shared DeepSeek-family weight loader +
#     attention backend handler.
#   - qwen2.py post-loop capture trap at line 652 is verbatim-copied for
#     Eagle3 aux hidden state capture (the `if end_layer in
#     layers_to_capture` block — known pitfall in wiki/pipeline.md).
#
# What is NEW code (not NSA-reusable):
#   - mHC Block (hc_pre, hc_post, hc_split_sinkhorn) — V4-specific
#   - FP4-aware Expert + V4Gate (hash routing first 3 layers + sqrtsoftplus
#     + noaux_tc) — V4-specific
#   - Per-layer compress_ratio dispatch (the WIN/CSA/HCA switch) — V4-specific
#   - deepseek_v4 chat-template registration (V4 has no Jinja template) —
#     V4-specific
#   - YaRN with dual rope_theta (10000 + 160000) — extension of V3.2 YaRN
#
# THIS FILE IS A SCAFFOLD. Class signatures + TODO markers + imports are in
# place. The actual implementation lands in subsequent commits on the
# eagle3/deepseek-v4 branch as each component is ported / written.
# ==============================================================================

"""Inference-only DeepSeek-V4 model — SCAFFOLD."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

# V3.2 NSA infrastructure that we will reuse for CSA + HCA.
# from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
# from sglang.srt.layers.attention.nsa.utils import (
#     cp_all_gather_rerange_output,
#     is_nsa_enable_prefill_cp,
# )
# from sglang.srt.configs.model_config import is_deepseek_nsa
#
# (Imports commented out during scaffolding to avoid circular import errors
# until the V4 model_type is registered. Will be uncommented as each
# component is ported.)

logger = logging.getLogger(__name__)


# ============================================================================
# Per-layer attention dispatch
# ============================================================================
#
# V4's `config.compress_ratios` array has one entry per main layer + one for
# the MTP block. Values are 0 (window-only), 4 (CSA), or 128 (HCA).
#
# Dispatch policy:
#   compress_ratio == 0   -> WindowOnlyAttention
#   compress_ratio == 4   -> CSAAttention (window + Indexer + Compressor)
#   compress_ratio == 128 -> HCAAttention (window + heavy Compressor, no Indexer)


class V4LayerAttentionType:
    """Enum-like marker for per-layer attention dispatch."""

    WINDOW_ONLY = "window_only"   # compress_ratio == 0
    CSA = "csa"                   # compress_ratio == 4
    HCA = "hca"                   # compress_ratio == 128

    @staticmethod
    def from_compress_ratio(compress_ratio: int) -> str:
        if compress_ratio == 0:
            return V4LayerAttentionType.WINDOW_ONLY
        if compress_ratio == 4:
            return V4LayerAttentionType.CSA
        if compress_ratio == 128:
            return V4LayerAttentionType.HCA
        raise ValueError(
            f"V4: unsupported compress_ratio={compress_ratio}. Expected 0, 4, or 128. "
            f"If config.compress_ratios contains a different value, the V4 release has "
            f"introduced a new layer type — extend V4LayerAttentionType to handle it."
        )


# ============================================================================
# Compressor — gated pooling over compress_ratio consecutive tokens
# ============================================================================
# Ported from V4's inference/model.py:Compressor. FP32-internal, BF16/FP8
# output. With compress_ratio=4 uses overlapping windows for smoother
# compression boundaries; with compress_ratio=128 uses non-overlapping.


class V4Compressor(nn.Module):
    """V4 KV-compression module.

    Ported 2026-04-30 from /tmp/v4-flash-meta/inference/model.py:Compressor
    (lines 279-377 of the V4 reference). Compresses KV cache via learned
    gated pooling over `compress_ratio` consecutive tokens.

    Two modes (per V4's per-layer attention dispatch):
      - `compress_ratio == 4` -> `overlap=True`. Two parallel compressions
        per chunk (offset by `ratio`) for smoother boundaries; uses
        `overlap_transform` to interleave.
      - `compress_ratio == 128` -> `overlap=False`. Standard non-overlap
        chunked compression.

    Decode-phase (start_pos > 0) uses incremental state buffers (`kv_state`,
    `score_state`) that accumulate per-token until a compression boundary
    fires (every `ratio` decode steps).

    Differences from the V4 reference impl, with TODO markers:
      - V4 reference uses parallel `Linear` (its own TP-aware class). We
        use `nn.Linear` here; TP-aware refactor at integration time.
        TODO(phase1-tp): swap wkv/wgate to `ColumnParallelLinear` once the
        TP layout is decided per architecture-notes.md.
      - V4 reference applies `apply_rotary_emb(kv[..., -rd:], freqs_cis)`
        in-place. We delegate to a `rotary_apply_fn` callable passed by
        the parent Attention layer (CSAAttention / HCAAttention). The
        sglang rotary embedding module owns the freqs_cis lifetime.
      - V4 reference applies `act_quant` / `fp4_act_quant` / `rotate_activation`
        for QAT simulation. These are TODO markers for the
        sgl-kernel quant path. For initial integration we run pure
        FP32/BF16; QAT-equivalent simulation lands in a follow-up.
        TODO(phase1-quant): wire sglang FP8 act_quant from
        sglang.srt.layers.quantization.fp8_kernel; add fp4_act_quant +
        Hadamard rotation kernels to sgl-kernel/csrc/.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        compress_ratio: int,
        rope_head_dim: int,
        max_batch_size: int,
        norm_eps: float = 1e-6,
        rotate: bool = False,
    ):
        super().__init__()
        self.dim = hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + int(self.overlap)

        # Learned compression parameters. ape = additive positional embedding
        # scoped to the compress window (per V4 reference line 294).
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # wkv / wgate stored in fp32 here for numerical convenience; the V4
        # checkpoint stores them in bf16 (V4 reference line 295-298). The
        # weight loader must up-cast on load.
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)

        # RMSNorm shim. Use sglang's layernorm module so the kernel matches
        # the rest of the sglang attention path.
        # TODO(phase1-norm): import lazily to avoid circular import; for now
        # use a thin functional implementation (matches V4 reference's RMSNorm
        # exactly: x.float().square().mean(-1).rsqrt() * weight).
        self.norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float32))
        self.norm_eps = norm_eps

        # Buffers wired by the parent Attention's __init__:
        #   kv_cache: the compressed-KV section of the parent's KV cache buffer.
        #   freqs_cis: the rotary frequencies (compress_rope_theta=160000 path).
        # Both are set lazily; see assertions in forward().
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

        # Decode-phase incremental state. With overlap: state[:, :ratio] is the
        # overlapping window, state[:, ratio:] the current window.
        self.register_buffer(
            "kv_state",
            torch.zeros(
                max_batch_size,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (max_batch_size, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        """Functional RMSNorm matching V4 reference exactly.
        TODO(phase1-norm): replace with sglang.srt.layers.layernorm.RMSNorm
        for kernel parity once sglang's RMSNorm accepts an external weight
        Parameter (current API constructs its own).
        """
        dtype = x.dtype
        x32 = x.float()
        var = x32.square().mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.norm_eps)
        return (self.norm_weight * x32).to(dtype)

    def overlap_transform(self, tensor: torch.Tensor, value: float = 0):
        """Overlap-mode reshape. Input [b, s, ratio, 2d] -> [b, s, 2*ratio, d].
        First ratio rows = previous chunk's overlap window, last ratio rows =
        current chunk. V4 reference lines 307-314."""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        rotary_apply_fn: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        """Compress the KV stream up to position start_pos+seqlen.

        Args:
            x: [bsz, seqlen, dim] input hidden states
            start_pos: token offset into the KV cache (0 = prefill, >0 = decode step)
            rotary_apply_fn: optional callable applied to the last `rope_head_dim`
                slice of the compressed KV. Signature: `fn(slice, freqs_cis) -> None`
                (in-place rotary). When None, no rotary applied (a TODO until the
                parent Attention wires its rotary path through).

        Returns:
            None when no compression boundary fired this call (decode steps
            between boundaries) or the compressed KV chunk written to kv_cache
            (prefill or boundary-firing decode step). Matches V4 reference
            behavior exactly.
        """
        assert self.kv_cache is not None, (
            "V4Compressor.forward called before parent Attention assigned kv_cache. "
            "The parent layer must set self.compressor.kv_cache = self.kv_cache[...] "
            "during its first forward."
        )
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        dtype = x.dtype

        # Compression runs in fp32 for numerical stability.
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
            # Prefill path. V4 reference lines 325-342.
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            # Decode path (start_pos > 0). V4 reference lines 343-359.
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        # RMSNorm + rotary on the rope-head-dim slice.
        kv = self._rmsnorm(kv.to(dtype))

        if rotary_apply_fn is not None:
            assert self.freqs_cis is not None, (
                "V4Compressor.freqs_cis must be assigned by the parent Attention "
                "before forward when rotary_apply_fn is provided."
            )
            if start_pos == 0:
                freqs_cis = self.freqs_cis[:cutoff:ratio]
            else:
                freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
            # In-place rotary on the rope_head_dim slice (V4 reference line 367).
            rotary_apply_fn(kv[..., -rd:], freqs_cis)

        # TODO(phase1-quant): apply Hadamard + FP4 quant when self.rotate is
        # True (Indexer path). Apply FP8 act_quant on the no-rope slice when
        # self.rotate is False (window/HCA path). Both are V4 QAT-simulation
        # operations; need sgl-kernel implementations. V4 reference lines
        # 368-372.

        # Write to compressed KV cache. Prefill writes the whole compressed
        # chunk; decode writes one compressed token at start_pos // ratio.
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


# ============================================================================
# CSAAttention — Compressed Sparse Attention (compress_ratio=4 + Indexer)
# ============================================================================


class CSAAttention(nn.Module):
    """V4 CSA layer: sliding window + Indexer + Compressor.

    Reuses V3.2 NSA Indexer (sglang.srt.layers.attention.nsa.nsa_indexer.Indexer)
    and the sparse_attn tilelang kernel.

    TODO(phase1-port): wire NSA Indexer with V4-specific config. The V3.2
    NSA path in deepseek_v2.py:1231 is the integration template. Differences:
      - V4 uses MQA (n_kv_heads=1), not MLA. Q/O paths are MLA-shaped (q_lora,
        grouped low-rank O); K/V paths are MQA-shaped (single shared head).
      - compress_ratio=4 (V4 default for CSA) — set via Indexer config.
      - V4 has dual rope_theta: 10000 for the window-only path, 160000 for
        the compressed KV path. NSA Indexer uses one rope_theta — needs
        extension or per-call override.
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        # TODO(phase1-port): full body
        raise NotImplementedError("CSAAttention scaffold — see TODO(phase1-port)")


# ============================================================================
# HCAAttention — Heavily Compressed Attention (compress_ratio=128, no Indexer)
# ============================================================================


class HCAAttention(nn.Module):
    """V4 HCA layer: sliding window + heavy Compressor (no Indexer).

    Uses the V3.2 NSA sparse_attn kernel with `topk_idxs` from
    `get_compress_topk_idxs` (deterministic stride selection) instead of the
    Indexer's learned-index topk.

    TODO(phase1-port): write the topk_idxs construction (port from V4
    reference inference/model.py:get_compress_topk_idxs lines 268-276) and
    wire it through the same sparse_attn kernel CSA uses. KV-cache layout
    differs from CSA: only the heavy-Compressor path, no Indexer kv_cache.
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        # TODO(phase1-port): full body
        raise NotImplementedError("HCAAttention scaffold — see TODO(phase1-port)")


# ============================================================================
# WindowOnlyAttention — sliding window only (compress_ratio=0)
# ============================================================================


class WindowOnlyAttention(nn.Module):
    """V4 window-only layer (used at layers 0, 1, 42).

    No compression branch. Pure sliding-window MLA-shaped Q/O + MQA-shaped K/V.
    rope_theta=10000, no YaRN scaling on this path.

    TODO(phase1-port): standard MQA + sliding window. The V3.2 NSA path
    has a window-only fallback (when is_deepseek_nsa is False); reuse that.
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        # TODO(phase1-port): full body
        raise NotImplementedError("WindowOnlyAttention scaffold — see TODO(phase1-port)")


# ============================================================================
# mHC (Manifold-Constrained Hyper-Connections) — V4-specific, NEW code
# ============================================================================


class V4HCBlock(nn.Module):
    """mHC residual-stream wrapping for a transformer block.

    V4-specific. NOT in V3.2 / V3 / V2. Maintains hc_mult=4 copies of the
    hidden state and mixes via Sinkhorn-normalized weights (hc_split_sinkhorn).

    The wrapping is per-sublayer: hc_pre folds 4 copies -> 1 input for
    the sublayer, runs the sublayer (attn or ffn), then hc_post expands 1
    -> 4 copies for the next sublayer.

    TODO(phase1-port): port from V4 reference inference/model.py:Block
    lines 647-700 (hc_pre + hc_post + the dispatch in forward()).

    TODO(phase1-kernel): write hc_split_sinkhorn kernel — V4 reference uses
    a triton kernel, ~150 lines in inference/kernel.py. Port to
    sgl-kernel/csrc/ with FlashInfer-style packaging.
    """

    def __init__(self, hidden_size: int, hc_mult: int = 4, hc_sinkhorn_iters: int = 20, hc_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        # TODO(phase1-port): allocate hc_attn_fn, hc_ffn_fn, hc_attn_base,
        # hc_ffn_base, hc_attn_scale, hc_ffn_scale parameters per V4 reference.
        raise NotImplementedError("V4HCBlock scaffold — see TODO(phase1-port + phase1-kernel)")


# ============================================================================
# V4Gate — hash routing on first 3 layers + score-based on rest
# ============================================================================


class V4Gate(nn.Module):
    """MoE gate. Hash routing on first `num_hash_layers=3` layers, score-based
    elsewhere. V4-specific (V3.2 used score-based throughout).

    Hash routing uses a `(vocab_size, num_experts_per_tok)=(129280, 6)` int32
    lookup table `tid2eid` indexed by the input token id. No softmax / no
    learned weights for the first 3 layers' routing decisions.

    Score-based routing uses sqrtsoftplus scoring + noaux_tc top-k (V3.2
    inheritance) + routed_scaling_factor=1.5.

    TODO(phase1-port): port from V4 reference inference/model.py:Gate
    lines 546-584. The score-based path can reuse logic from
    deepseek_v2.py:DeepseekV2MoE — only the hash branch is new.
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.is_hash = layer_id < getattr(config, "num_hash_layers", 0)
        # TODO(phase1-port): full body
        raise NotImplementedError("V4Gate scaffold — see TODO(phase1-port)")


# ============================================================================
# DeepseekV4MoE — FP4-aware Mixture of Experts
# ============================================================================


class DeepseekV4MoE(nn.Module):
    """V4 MoE: 256 routed experts top-6 + 1 shared expert.

    Expert weights are FP4 (`torch.float4_e2m1fn_x2`) with FP8 (e8m0fnu) scales,
    quantized in 32-element blocks. Non-expert weights are FP8 (e4m3fn) with
    e8m0fnu scales in 128x128 blocks.

    TODO(phase1-port): adapt deepseek_v2.py:DeepseekV2MoE for FP4 expert weight
    handling. The Gate needs to be the V4Gate above (with hash routing branch).
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        # TODO(phase1-port): full body
        raise NotImplementedError("DeepseekV4MoE scaffold — see TODO(phase1-port)")


# ============================================================================
# DeepseekV4DecoderLayer — wraps Attention + MoE + mHC
# ============================================================================


class DeepseekV4DecoderLayer(nn.Module):
    """One transformer block. Per-layer attention dispatch on
    `config.compress_ratios[layer_id]`. mHC residuals wrap both sublayers.
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id

        # Per-layer attention dispatch
        compress_ratio = config.compress_ratios[layer_id]
        attn_type = V4LayerAttentionType.from_compress_ratio(compress_ratio)
        if attn_type == V4LayerAttentionType.WINDOW_ONLY:
            self.attn = WindowOnlyAttention(layer_id, config)
        elif attn_type == V4LayerAttentionType.CSA:
            self.attn = CSAAttention(layer_id, config)
        elif attn_type == V4LayerAttentionType.HCA:
            self.attn = HCAAttention(layer_id, config)

        self.ffn = DeepseekV4MoE(layer_id, config)
        self.hc = V4HCBlock(
            hidden_size=config.hidden_size,
            hc_mult=getattr(config, "hc_mult", 4),
            hc_sinkhorn_iters=getattr(config, "hc_sinkhorn_iters", 20),
            hc_eps=getattr(config, "hc_eps", 1e-6),
        )
        # TODO(phase1-port): attn_norm, ffn_norm RMSNorm

    def forward(self, x, *args, **kwargs):
        # TODO(phase1-port): mHC pre/post wrapping per V4 reference Block.forward
        raise NotImplementedError("DeepseekV4DecoderLayer.forward scaffold — see TODO(phase1-port)")


# ============================================================================
# DeepseekV4ForCausalLM — the top-level entrypoint discovered by sglang
# ============================================================================
#
# NOTE: sglang's models registry (sglang/python/sglang/srt/models/registry.py)
# auto-discovers model classes via pkgutil walk and matches
# `hf_config.architectures[0]` to the class name. V4-Flash's config.json
# sets `architectures: ["DeepseekV4ForCausalLM"]`, so this class name MUST
# match. Do NOT rename without updating downstream registry usage.


class DeepseekV4ForCausalLM(nn.Module):
    """DeepSeek-V4 inference entrypoint."""

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__()
        self.config = config

        # Store Eagle3 capture config; populated by set_eagle3_layers_to_capture.
        self._eagle3_layers_to_capture: List[int] = []
        self._enable_return_hidden_states: bool = False

        # TODO(phase1-port): build embed, layers, norm, lm_head
        # self.embed = nn.Embedding(config.vocab_size, config.hidden_size, ...)
        # self.layers = nn.ModuleList([
        #     DeepseekV4DecoderLayer(i, config) for i in range(config.num_hidden_layers)
        # ])
        # self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # self.lm_head = ...
        # MTP block per config.num_nextn_predict_layers (default 1) — for Eagle3,
        # we drop MTP at inference time; load weights but don't run them.
        # logger.info(...)

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]) -> None:
        """Eagle3 hook: configure which transformer layer indices' aux hidden
        states should be captured + concatenated into the auxiliary tensor
        returned alongside the main output.

        For V4-Flash, the proposal in architecture-notes.md is `[1, 21, 41]`.
        These three layers' hidden states are concatenated along the feature
        dim into shape `(B, T, sum_of_per_layer_hidden_sizes)`. For V4,
        each layer has hidden_size=4096, so the aux tensor shape is
        `(B, T, 12288)`.

        TODO(phase1-port): apply the post-loop capture trap from qwen2.py:652
        — when `end_layer in layers_to_capture`, capture INSIDE the loop's
        last iteration to avoid the off-by-one bug documented in
        wiki/pipeline.md known-pitfalls.
        """
        self._eagle3_layers_to_capture = list(layers_to_capture)
        logger.info(
            "DeepseekV4ForCausalLM.set_eagle3_layers_to_capture: %s",
            self._eagle3_layers_to_capture,
        )

    @property
    def enable_return_hidden_states(self) -> bool:
        return self._enable_return_hidden_states

    @enable_return_hidden_states.setter
    def enable_return_hidden_states(self, value: bool) -> None:
        self._enable_return_hidden_states = bool(value)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        # TODO(phase1-port): full forward with embed -> hc-expand -> layers ->
        # hc-head -> lm_head. Emit aux hidden states when
        # self._enable_return_hidden_states is True, capturing on layers
        # in self._eagle3_layers_to_capture (apply qwen2.py:652 post-loop trap
        # for layer indices == num_hidden_layers - 1 if present).
        raise NotImplementedError("DeepseekV4ForCausalLM.forward scaffold — see TODO(phase1-port)")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Adapt deepseek_common/deepseek_weight_loader for FP4-aware loading.

        TODO(phase1-port): need to handle FP4 (`torch.float4_e2m1fn_x2`) expert
        weight tensors with their E8M0 scales. The V3.2 weight loader is
        BF16/FP8 aware but not FP4.
        """
        raise NotImplementedError("DeepseekV4ForCausalLM.load_weights scaffold — see TODO(phase1-port)")


# ============================================================================
# Module-level export — explicit so the registry's pkgutil walk picks it up
# ============================================================================

EntryClass = DeepseekV4ForCausalLM
