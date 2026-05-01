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
import math
from functools import lru_cache
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
# Module-level helpers ported from V4 reference inference/model.py
# ============================================================================
# These five helpers are direct ports. They will be replaced with sglang-native
# equivalents at integration time:
#   - `precompute_freqs_cis_yarn` -> sglang's RotaryEmbedding with YaRN scaling
#   - `apply_rotary_emb` -> sglang's apply_rotary_pos_emb (after kernel parity check)
#   - `get_window_topk_idxs` / `get_compress_topk_idxs` -> sglang NSA equivalent
#     where one exists; otherwise stay here
#
# Keeping them as standalone functions matches the V4 reference exactly and
# makes numerical agreement testing trivial in the unit-test phase.


@lru_cache(maxsize=4)
def precompute_freqs_cis_yarn(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """Precompute complex exponentials for rotary embeddings with YaRN scaling.

    Direct port of V4 reference `precompute_freqs_cis` (lines 199-229).
    When `original_seq_len > 0`, applies YaRN frequency interpolation with a
    smooth linear ramp between `beta_fast` and `beta_slow` correction ranges.

    V4-Flash uses two different invocations:
      - sliding-window-only path: original_seq_len=0, base=10000 (no YaRN)
      - compressed-KV path: original_seq_len=65536, base=160000, factor=16
        (YaRN scaling from 64K to 1M)

    Args:
        dim: rotary head dim (= rope_head_dim, default 64 for V4)
        seqlen: max sequence length to precompute up to (= max_position_embeddings)
        original_seq_len: pre-YaRN context length; 0 disables YaRN
        base: rope_theta (10000 for window-only, 160000 for compressed)
        factor: YaRN scale factor (16 for V4-Flash)
        beta_fast: high-frequency YaRN cutoff (32 for V4)
        beta_slow: low-frequency YaRN cutoff (1 for V4)

    Returns:
        Complex tensor of shape `[seqlen, dim // 2]` containing freqs_cis.
    """

    def find_correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(
        low_rot: int, high_rot: int, dim: int, base: float, max_seq_len: int
    ) -> Tuple[int, int]:
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(low: float, high: float, dim: int) -> torch.Tensor:
        if low == high:
            high = high + 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb_v4(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary positional embeddings in-place. Direct port of V4 reference
    `apply_rotary_emb` (lines 232-244).

    Uses conjugate for inverse (de-rotation, used after attention output).
    Suffix `_v4` to avoid clashing with sglang's existing apply_rotary_pos_emb;
    will be unified when the kernel-parity check is done.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@lru_cache(maxsize=1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """Build the per-token top-k index matrix for sliding-window attention.

    Direct port of V4 reference (lines 254-265). `lru_cache` keyed on the
    args because the result depends only on (window_size, bsz, seqlen,
    start_pos) — not on token content.
    """
    if start_pos >= window_size - 1:
        start_pos = start_pos % window_size
        matrix = torch.cat(
            [torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(maxsize=2)
def get_compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int
) -> torch.Tensor:
    """Build the per-token top-k index matrix for the compressed-KV branch.

    Direct port of V4 reference (lines 268-276). Used by HCA layers
    (compress_ratio=128) where the top-k is deterministic stride selection
    rather than learned-index (CSA's Indexer).
    """
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def sparse_attn_v4(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention kernel for V4 CSA + HCA layers.

    TODO(phase1-kernel): wire to sglang's NSA tilelang kernel
    (sglang/srt/layers/attention/nsa/tilelang_kernel.py). The V3.2 NSA path
    uses the same shape contract: Q [b, s, h, d], KV [b, k, d] with single
    shared K/V head (MQA), topk_idxs [b, s, k_per_query] selecting from KV
    along the k axis, attn_sink [h] adding a per-head sink-token
    contribution, softmax_scale = head_dim ** -0.5.

    Until wired, this raises so any actual forward call is caught loudly
    instead of silently returning garbage.
    """
    raise NotImplementedError(
        "sparse_attn_v4: wire to sglang/srt/layers/attention/nsa/tilelang_kernel.py "
        "(see TODO(phase1-kernel))."
    )


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
# V4Attention — single class for all three attention modes (matches V4 reference)
# ============================================================================
#
# The V4 reference impl (inference/model.py:Attention) is one class that
# dispatches per-instance on `compress_ratio = config.compress_ratios[layer_id]`:
#   - compress_ratio == 0   -> window-only (no Compressor, no Indexer)
#   - compress_ratio == 4   -> CSA (Compressor + Indexer)
#   - compress_ratio == 128 -> HCA (Compressor, no Indexer; uses
#                                  get_compress_topk_idxs)
#
# Following the reference exactly is the cheapest path to numerical agreement.
# Splitting into three classes was a scaffolding placeholder — collapsed here
# 2026-04-30 once the V4 reference Attention.__init__ + .forward were ported.


class V4Attention(nn.Module):
    """V4 attention layer, single class for window-only / CSA / HCA modes.

    Direct port of /tmp/v4-flash-meta/inference/model.py:Attention (lines
    436-543). Q path uses MLA-shaped low-rank projection (wq_a -> q_norm
    -> wq_b, q_lora_rank=1024); O path uses grouped low-rank projection
    (wo_a/wo_b with o_groups=8, o_lora_rank=1024); K/V path uses MQA
    (single shared K/V head fed via wkv: dim -> head_dim).

    Per-layer dispatch via `compress_ratio`:
      - 0: window-only attention. No Compressor, no Indexer. Uses
           get_window_topk_idxs for the topk index matrix.
      - 4: CSA. Compressor + Indexer (V3.2 NSA Indexer). topk_idxs is the
           concatenation of window indices and Indexer-selected compressed
           KV indices.
      - 128: HCA. Compressor only (no Indexer). topk_idxs is the
             concatenation of window indices and deterministic stride
             indices from get_compress_topk_idxs.

    KV cache layout: `kv_cache[:bsz, :window_size]` holds the sliding
    window; `kv_cache[:bsz, window_size:]` holds compressed KV when
    compress_ratio is non-zero. The Compressor's `kv_cache` member aliases
    the compressed section (assigned lazily in forward()).

    Differences from V4 reference, all flagged inline as TODO markers:

    - Linear/ColumnParallelLinear/RowParallelLinear: V4 reference's parallel
      Linear classes are TP-aware. We use `nn.Linear` here pending sglang
      TP-aware refactor (TODO(phase1-tp)). On a single GPU with world_size=1
      the behavior is identical.
    - sparse_attn: stubbed (raises) until wired to sglang NSA tilelang
      (TODO(phase1-kernel)).
    - act_quant calls (V4 reference line 506) skipped pending FP8 quant
      wiring (TODO(phase1-quant)).
    - Indexer: V3.2 sglang Indexer at sglang.srt.layers.attention.nsa.nsa_indexer
      will plug into self.indexer when compress_ratio == 4 (TODO(phase1-nsa)).
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        # TP-aware: divide n_heads by world size; with world_size=1 this is identity.
        self.n_local_heads = self.n_heads  # TODO(phase1-tp): // world_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", config.head_dim)
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups  # TODO(phase1-tp): // world_size
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_id]
        self.eps = config.rms_norm_eps

        # Attention sink (per-head; V4 reference line 456). Float32.
        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))

        # Q path: low-rank projection (q_lora_rank=1024).
        # TODO(phase1-tp): wq_a is replicated, wq_b is column-parallel.
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        # Use functional RMSNorm matching V4 reference; will swap to sglang's
        # RMSNorm once external-weight API lands (TODO(phase1-norm)).
        self.q_norm_weight = nn.Parameter(torch.ones(self.q_lora_rank, dtype=torch.float32))
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # K/V path: MQA (single shared head). wkv: dim -> head_dim.
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float32))

        # O path: grouped low-rank projection.
        # wo_a is column-parallel: (n_heads * head_dim / n_groups, n_groups * o_lora_rank)
        # wo_b is row-parallel: (n_groups * o_lora_rank, dim)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        self.softmax_scale = self.head_dim ** -0.5

        # Optional Compressor + Indexer per compress_ratio.
        if self.compress_ratio:
            self.compressor = V4Compressor(
                hidden_size=self.dim,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                rope_head_dim=self.rope_head_dim,
                max_batch_size=getattr(config, "max_batch_size", 4),
                norm_eps=self.eps,
                rotate=(self.compress_ratio == 4),  # CSA uses Hadamard rotation; HCA doesn't
            )
            if self.compress_ratio == 4:
                # TODO(phase1-nsa): plug in sglang's V3.2 NSA Indexer here.
                # `self.indexer = NSAIndexer(...)` once configured for V4 args.
                self.indexer = None
            else:
                # HCA: no learned Indexer, deterministic stride selection.
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # KV cache buffer. Window section + optional compressed section.
        max_seq_len = config.max_position_embeddings
        max_batch_size = getattr(config, "max_batch_size", 4)
        kv_cache_size = self.window_size + (
            max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, kv_cache_size, self.head_dim),
            persistent=False,
        )

        # Rotary frequencies. Two regimes per V4 reference lines 475-481:
        #   compress_ratio != 0 -> YaRN with compress_rope_theta (160000) from
        #                          original_seq_len (65536) up to max_seq_len (1M)
        #   compress_ratio == 0 -> no YaRN, base rope_theta (10000)
        if self.compress_ratio:
            original_seq_len = config.rope_scaling["original_max_position_embeddings"]
            rope_theta = config.compress_rope_theta
            rope_factor = config.rope_scaling["factor"]
        else:
            original_seq_len = 0
            rope_theta = config.rope_theta
            rope_factor = 1.0
        beta_fast = config.rope_scaling.get("beta_fast", 32)
        beta_slow = config.rope_scaling.get("beta_slow", 1)
        freqs_cis = precompute_freqs_cis_yarn(
            self.rope_head_dim,
            max_seq_len,
            original_seq_len,
            rope_theta,
            rope_factor,
            beta_fast,
            beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    # -----------------------------------------------------------------
    # RMSNorm helpers (functional; matches V4 reference exactly)
    # -----------------------------------------------------------------

    def _q_norm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        var = x32.square().mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.q_norm_weight * x32).to(dtype)

    def _kv_norm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        var = x32.square().mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.kv_norm_weight * x32).to(dtype)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """V4 attention forward. Direct port of V4 reference (lines 484-543)."""
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Lazy wiring: parent Compressor/Indexer get the kv_cache + freqs_cis on first call.
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # ---- Q ----
        qr = q = self._q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        # Per-head RMS scaling (V4 reference line 498).
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb_v4(q[..., -rd:], freqs_cis)

        # ---- Window K/V ----
        kv = self.wkv(x)
        kv = self._kv_norm(kv)
        apply_rotary_emb_v4(kv[..., -rd:], freqs_cis)
        # TODO(phase1-quant): act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for
        # positional precision. Skipped pending FP8 quant wiring.

        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                # CSA path: learned-index topk via the Indexer.
                # TODO(phase1-nsa): self.indexer(x, qr, start_pos, offset)
                raise NotImplementedError(
                    "V4Attention CSA path: Indexer not yet wired (TODO(phase1-nsa))."
                )
            else:
                # HCA path: deterministic stride topk.
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
                topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # ---- Compress KV + sparse attention ----
        if start_pos == 0:
            # Prefill: write the last `win` tokens into the window cache. If
            # seqlen > win, the older tokens are dropped (sliding window).
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = (
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
                )
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos, rotary_apply_fn=apply_rotary_emb_v4)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn_v4(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            # Decode: write the single new token at start_pos % win.
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos, rotary_apply_fn=apply_rotary_emb_v4)
            o = sparse_attn_v4(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)

        # Inverse rotary on output (V4 reference line 534).
        apply_rotary_emb_v4(o[..., -rd:], freqs_cis, inverse=True)

        # ---- O projection (grouped low-rank) ----
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum for better perf,
        # but using BF16 here for simplicity (matches V4 reference comment).
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


# Keep backward-compat aliases for the three names; their __init__ now
# delegates to V4Attention. This lets external code that imported the old
# names continue to work during the porting window.
class CSAAttention(V4Attention):
    """Alias preserved for backward-compat. Use V4Attention directly going forward."""
    pass


class HCAAttention(V4Attention):
    """Alias preserved for backward-compat. Use V4Attention directly going forward."""
    pass


class WindowOnlyAttention(V4Attention):
    """Alias preserved for backward-compat. Use V4Attention directly going forward."""
    pass


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

        # Single-class attention. Per-layer behavior is encoded in
        # config.compress_ratios[layer_id] which the V4Attention __init__
        # consumes directly.
        self.attn = V4Attention(layer_id, config)

        # MoE feed-forward. V4Gate inside DeepseekV4MoE checks
        # layer_id < config.num_hash_layers for the hash-routing branch.
        self.ffn = DeepseekV4MoE(layer_id, config)

        # mHC residual-stream wrapping. Wraps both attn and ffn separately.
        self.hc = V4HCBlock(
            hidden_size=config.hidden_size,
            hc_mult=getattr(config, "hc_mult", 4),
            hc_sinkhorn_iters=getattr(config, "hc_sinkhorn_iters", 20),
            hc_eps=getattr(config, "hc_eps", 1e-6),
        )

        # Pre-sublayer RMSNorms (matching V4 reference Block.__init__ lines
        # 658-659). Functional impl pending sglang RMSNorm external-weight API.
        self.attn_norm_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.float32))
        self.ffn_norm_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.float32))
        self.norm_eps = config.rms_norm_eps

    def _norm(self, x: torch.Tensor, weight: nn.Parameter) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        var = x32.square().mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.norm_eps)
        return (weight * x32).to(dtype)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """V4 transformer block forward with mHC residuals.

        Direct port of V4 reference Block.forward (lines 688-700). Each
        sublayer is wrapped: hc_pre folds 4 copies -> 1 input, sublayer runs,
        hc_post expands 1 -> 4 copies for the next sublayer.

        TODO(phase1-port): V4HCBlock.forward / hc_pre / hc_post need to land
        before this works. Until then this raises with a clear message.
        """
        # Attention sublayer (mHC-wrapped)
        residual = x
        x_pre, post, comb = self.hc.hc_pre(x, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base)
        x_pre = self._norm(x_pre, self.attn_norm_weight)
        x_pre = self.attn(x_pre, start_pos)
        x = self.hc.hc_post(x_pre, residual, post, comb)

        # FFN sublayer (mHC-wrapped)
        residual = x
        x_pre, post, comb = self.hc.hc_pre(x, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base)
        x_pre = self._norm(x_pre, self.ffn_norm_weight)
        x_pre = self.ffn(x_pre, input_ids)
        x = self.hc.hc_post(x_pre, residual, post, comb)
        return x


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
