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


def hc_split_sinkhorn_v4(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch port of V4 reference `hc_split_sinkhorn` (kernel.py
    lines 372-438).

    Splits the `mixes` tensor of shape `[..., (2+hc)*hc]` into three pieces:
      pre  = sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps        -> [..., hc]
      post = 2 * sigmoid(mixes[..., hc:2*hc] * scale[1] + base[hc:2*hc])  -> [..., hc]
      comb = Sinkhorn-normalized matrix from mixes[..., 2*hc:]            -> [..., hc, hc]

    Sinkhorn algorithm (matches the kernel exactly):
      1. comb_logits = mixes[..., 2*hc:].reshape(..., hc, hc) * scale[2] + base[2*hc:].reshape(hc, hc)
      2. comb = softmax(comb_logits, dim=-1) + eps
      3. comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
      4. Repeat (sinkhorn_iters - 1) times:
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    TODO(phase1-kernel): port to a tilelang kernel matching the V4 reference
    `hc_split_sinkhorn_kernel` in /tmp/v4-flash-meta/inference/kernel.py.
    The pure-PyTorch impl here is correct but slower; the kernel runs the
    Sinkhorn loop entirely on-chip with `T.alloc_fragment` storage. Add to
    sgl-kernel/csrc/ alongside the FP4 GEMM kernel.

    Args:
        mixes: [..., (2+hc_mult)*hc_mult] FP32
        hc_scale: [3] FP32 — broadcast scale for pre/post/comb logits
        hc_base: [(2+hc_mult)*hc_mult] FP32 — additive bias
        hc_mult: number of HC copies (V4-Flash uses 4)
        sinkhorn_iters: total Sinkhorn iterations (V4-Flash uses 20)
        eps: numerical stability

    Returns:
        (pre, post, comb) with shapes:
            pre: [..., hc_mult]
            post: [..., hc_mult]
            comb: [..., hc_mult, hc_mult]
    """
    hc = hc_mult
    leading_shape = mixes.shape[:-1]

    # Slice and apply scale + base.
    pre_logits = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
    post_logits = mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    comb_logits = (
        mixes[..., 2 * hc :].reshape(*leading_shape, hc, hc) * hc_scale[2]
        + hc_base[2 * hc :].reshape(hc, hc)
    )

    pre = torch.sigmoid(pre_logits) + eps
    post = 2 * torch.sigmoid(post_logits)

    # Initial: row-softmax + eps, then divide column sums.
    comb = F.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    # Subsequent (sinkhorn_iters - 1) iterations: alternate row/col normalization.
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


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
    """mHC (Manifold-Constrained Hyper-Connections) residual-stream wrapping.

    V4-specific NEW code. NOT in V3.2 / V3 / V2. Maintains `hc_mult=4` copies
    of the hidden state per token and mixes them via Sinkhorn-normalized
    weights computed by `hc_split_sinkhorn_v4`.

    Per-sublayer wrapping (called twice per Block, once for attn, once for ffn):
      hc_pre: folds 4 copies -> 1 input. Returns (y, post, comb) where:
              y    = sum_i pre[i] * x[i]                 [b, s, d]
              post = sigmoid-derived per-copy weights    [b, s, hc]
              comb = Sinkhorn-doubly-stochastic mixer    [b, s, hc, hc]
      hc_post: expands 1 -> 4 copies for the next sublayer.
              y[i] = post[i] * x + sum_j comb[i,j] * residual[j]

    Direct port of V4 reference `Block` (inference/model.py lines 647-700).
    The V4 reference packs attn + ffn + their hc parameters into a single
    `Block` class; our scaffold splits the hc parameters into `V4HCBlock` and
    keeps attn / ffn on `DeepseekV4DecoderLayer`. Functionally identical.

    Parameter shapes (V4 reference lines 660-671, all FP32):
      hc_attn_fn, hc_ffn_fn:  [(2+hc_mult)*hc_mult, hc_mult*hidden_size]
      hc_attn_base, hc_ffn_base: [(2+hc_mult)*hc_mult]
      hc_attn_scale, hc_ffn_scale: [3]
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        mix_hc = (2 + hc_mult) * hc_mult  # = 24 for hc_mult=4
        hc_dim = hc_mult * hidden_size

        # All HC parameters are FP32 (V4 reference uses set_dtype(torch.float32)
        # context manager around the parameter allocation).
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fold hc_mult copies into 1 weighted-sum input for the sublayer.

        Direct port of V4 reference Block.hc_pre (lines 673-681).

        Args:
            x: [b, s, hc_mult, d] hidden state with hc_mult copies per token
            hc_fn: [mix_hc, hc_mult*d] linear projection to the mixes vector
            hc_scale: [3] additional scale on the three Sinkhorn input groups
            hc_base: [mix_hc] additive bias on the mixes

        Returns:
            (y, post, comb):
                y: [b, s, d] folded input for the sublayer
                post: [b, s, hc_mult] post-weights for hc_post
                comb: [b, s, hc_mult, hc_mult] combination matrix for hc_post
        """
        shape, dtype = x.size(), x.dtype
        # Flatten the hc_mult copies into a single feature dim (b, s, hc*d), fp32.
        x_flat = x.flatten(2).float()
        # Per-token RMS scale (V4 reference line 677).
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn_v4(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        # y = sum over hc copies, weighted by pre.
        # x.view(shape) is [b, s, hc_mult, d]. pre.unsqueeze(-1) is [b, s, hc_mult, 1].
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand the sublayer output back to hc_mult copies.

        Direct port of V4 reference Block.hc_post (lines 683-686).

        Args:
            x: [b, s, d] sublayer output
            residual: [b, s, hc_mult, d] pre-sublayer hidden state (hc copies)
            post: [b, s, hc_mult] from hc_pre
            comb: [b, s, hc_mult, hc_mult] from hc_pre

        Returns:
            y: [b, s, hc_mult, d] post-sublayer hidden state with hc copies
        """
        # post.unsqueeze(-1):                [b, s, hc_mult, 1]
        # x.unsqueeze(-2):                   [b, s, 1, d]
        # comb.unsqueeze(-1):                [b, s, hc_mult, hc_mult, 1]
        # residual.unsqueeze(-2):            [b, s, 1, hc_mult, d]
        # comb * residual sums over hc_mult: [b, s, hc_mult, d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)


# ============================================================================
# V4Gate — hash routing on first 3 layers + score-based on rest
# ============================================================================


class V4Gate(nn.Module):
    """MoE gate. Hash routing on first `num_hash_layers=3` layers, score-based
    elsewhere. Direct port of V4 reference Gate (inference/model.py lines
    546-584).

    Hash routing (first 3 layers, V4-specific): the `tid2eid` lookup table
    [vocab_size, num_experts_per_tok]=(129280, 6) int32 maps each token id
    to a fixed set of 6 experts. No softmax, no learned weights for the
    routing decision (the lookup table is loaded but `requires_grad=False`).
    The routing weights still come from the score-based path even on hash
    layers — the `weights = original_scores.gather(1, indices)` step uses
    the score-based scores; only the `indices` come from the hash table.

    Score-based routing (layers 3+): sqrtsoftplus(weight @ x) -> +bias ->
    top-k -> gather original (pre-bias) scores -> normalize -> scale.
    Matches V3.2's noaux_tc routing semantics.
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id

        self.dim = config.hidden_size
        self.topk = config.num_experts_per_tok
        # V4 uses scoring_func=sqrtsoftplus; V2/V3 also support softmax/sigmoid.
        self.score_func = getattr(config, "scoring_func", "sqrtsoftplus")
        self.route_scale = getattr(config, "routed_scaling_factor", 1.0)
        self.is_hash = layer_id < getattr(config, "num_hash_layers", 0)

        # Routing projection: x [B*T, dim] -> scores [B*T, n_routed_experts].
        # V4 reference stores this as `weight` directly (not nn.Linear) so the
        # weight loader can match the checkpoint key naming.
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )

        if self.is_hash:
            # tid2eid: lookup table from token id -> [num_experts_per_tok]
            # expert ids. Loaded from checkpoint, frozen.
            self.tid2eid = nn.Parameter(
                torch.empty(config.vocab_size, self.topk, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            # Score-bias for expert-selection top-k (does not affect routing
            # weights themselves; matches V3.2 noaux_tc semantics).
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights + expert indices for each token.

        Args:
            x: [B*T, dim] token-flattened hidden state
            input_ids: [B*T] token ids; required only when `is_hash` is True.

        Returns:
            (weights, indices):
                weights: [B*T, topk] routing weights (scaled, normalized)
                indices: [B*T, topk] expert indices to route to
        """
        # Routing projection in fp32 for numerical stability.
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus (V4 default)
            scores = F.softplus(scores).sqrt()
        original_scores = scores

        # Bias shifts scores for top-k selection only (V4 reference line 575).
        if self.bias is not None:
            scores = scores + self.bias

        if self.is_hash:
            assert input_ids is not None, "Hash-routing layers require input_ids"
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]

        # Gather pre-bias scores at the selected indices.
        weights = original_scores.gather(1, indices.long())
        # Normalize when not softmax (V4 reference line 581-582).
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class V4Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN. Direct port of V4 reference Expert
    (inference/model.py lines 587-606).

    V4 expert weights are FP4 (`torch.float4_e2m1fn_x2`) with E8M0 scales for
    routed experts; shared expert is BF16. Computation runs in FP32 inside
    silu(gate) * up for stability, then casts back to input dtype before w2.

    TODO(phase1-fp4): swap nn.Linear to V4-aware FP4 Linear that handles
    `torch.float4_e2m1fn_x2` weight storage with E8M0 scales. Until then,
    we use nn.Linear with bf16 weights and rely on the weight loader to
    dequantize FP4 -> BF16 on load. That's slow + memory-heavy at runtime
    but functionally correct for the first integration test.
    """

    def __init__(
        self, dim: int, inter_dim: int, swiglu_limit: float = 0.0, dtype=None
    ):
        super().__init__()
        # nn.Linear here; FP4 path is TODO(phase1-fp4).
        # The reference V4 Linear class accepts `dtype=torch.float4_e2m1fn_x2`
        # and stores weights in packed FP4. We use bf16 here (the loader
        # dequantizes on load).
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """SwiGLU expert forward. V4 ref lines 596-606."""
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


# ============================================================================
# DeepseekV4MoE — FP4-aware Mixture of Experts
# ============================================================================


class DeepseekV4MoE(nn.Module):
    """V4 MoE: 256 routed experts top-6 + 1 shared expert.

    Direct port of V4 reference MoE (inference/model.py lines 609-644).

    Routing: V4Gate above. Hash routing on first num_hash_layers=3 layers
    (indices from tid2eid lookup); score-based routing on the rest
    (sqrtsoftplus + topk).

    Expert weights: FP4 (`torch.float4_e2m1fn_x2`) with FP8 (e8m0fnu) scales,
    quantized in 32-element blocks along K (reduce dim). Shared expert is
    BF16 / FP8 (matching the rest of the model's non-expert quantization).

    TP behavior: V4 reference shards experts across world_size. With
    world_size=1 this is pass-through. TODO(phase1-tp): apply
    expert-parallel sharding when integrated into sglang's TP runtime.
    """

    def __init__(self, layer_id: int, config: PretrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size

        # TP-aware: with world_size=1, all experts are local.
        self.n_routed_experts = config.n_routed_experts
        self.n_local_experts = config.n_routed_experts  # TODO(phase1-tp): // world_size
        self.n_activated_experts = config.num_experts_per_tok
        self.experts_start_idx = 0  # TODO(phase1-tp): rank * n_local_experts
        self.experts_end_idx = self.n_local_experts
        n_shared_experts = getattr(config, "n_shared_experts", 1)
        assert n_shared_experts == 1, (
            f"V4 expects exactly 1 shared expert per layer; config has {n_shared_experts}"
        )

        # Routing gate (hash-or-score per layer_id < num_hash_layers).
        self.gate = V4Gate(layer_id, config)

        # Routed experts. V4 reference stores None for non-local-rank experts
        # to save memory; with world_size=1 this is just a list of all experts.
        # TODO(phase1-fp4): wire FP4 expert dtype handling. For now bf16 storage.
        moe_inter_dim = config.moe_intermediate_size
        swiglu_limit = getattr(config, "swiglu_limit", 0.0)
        self.experts = nn.ModuleList(
            [
                V4Expert(self.dim, moe_inter_dim, swiglu_limit=swiglu_limit)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )

        # Shared expert (always-on, every token).
        self.shared_experts = V4Expert(self.dim, moe_inter_dim, swiglu_limit=swiglu_limit)

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """V4 MoE forward. Direct port of V4 reference MoE.forward (lines 629-644).

        Args:
            x: [B, T, dim] hidden state
            input_ids: [B, T] token ids (required for hash-routing layers)

        Returns:
            y: [B, T, dim] MoE output (sum of routed experts + shared expert)
        """
        shape = x.size()
        x_flat = x.view(-1, self.dim)
        flat_input_ids = input_ids.flatten() if input_ids is not None else None
        weights, indices = self.gate(x_flat, flat_input_ids)

        # Accumulate routed expert outputs.
        y = torch.zeros_like(x_flat, dtype=torch.float32)

        # V4 reference uses bincount + per-expert iteration. This is fine for
        # bf16 inference; sglang has a fused MoE path for higher throughput
        # we'll wire later (TODO(phase1-fused-moe)).
        counts = torch.bincount(
            indices.flatten().long(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] = y[idx] + expert(x_flat[idx], weights[idx, top, None])

        # TP all_reduce stub (world_size=1 -> noop). TODO(phase1-tp).
        # if world_size > 1: dist.all_reduce(y)

        # Shared expert: always-on; runs on every token.
        y = y + self.shared_experts(x_flat)
        return y.type_as(x).view(shape)


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
    """DeepSeek-V4 inference entrypoint.

    Direct port of V4 reference Transformer (inference/model.py lines 769-809)
    with Eagle3 aux-hidden-state capture grafted on per the qwen2.py:652
    post-loop trap pattern (`wiki/pipeline.md` known-pitfall).

    Per CLAUDE.md rule #12 + architecture-notes.md "Eagle3 vs native MTP",
    the MTP block is loaded for weight-key compatibility but NOT executed
    at Eagle3 inference time. The Eagle3 draft head replaces it.
    """

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__()
        self.config = config

        # Eagle3 capture config; populated by set_eagle3_layers_to_capture.
        self._eagle3_layers_to_capture: List[int] = []
        self._enable_return_hidden_states: bool = False

        # Embedding + main transformer trunk.
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(i, config) for i in range(config.num_hidden_layers)]
        )

        # Final RMSNorm before head (V4 reference line 787).
        self.final_norm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=torch.float32)
        )
        self.norm_eps = config.rms_norm_eps

        # Top-level mHC head reduction parameters (V4 reference lines 794-799).
        # These fold the [b, s, hc_mult, d] hidden state into [b, s, d] before lm_head.
        hc_mult = getattr(config, "hc_mult", 4)
        self.hc_mult = hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_eps = getattr(config, "hc_eps", 1e-6)

        # LM head. V4 reference stores in fp32 for logit precision.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP blocks. Per rule #12, loaded but NOT run at Eagle3 inference;
        # the Eagle3 draft head replaces them. Allocating the module list
        # so the weight loader can write the MTP weight keys without error.
        # TODO(phase1-port): MTPBlock body (mirrors DeepseekV4DecoderLayer +
        # e_proj, h_proj, enorm, hnorm, head_fn/base/scale per V4 ref lines
        # 738-766). Skipped tonight because we don't run MTP for Eagle3.
        n_mtp = getattr(config, "num_nextn_predict_layers", 0)
        if n_mtp > 0:
            logger.info(
                "DeepseekV4ForCausalLM: %d MTP block(s) detected in config "
                "(num_nextn_predict_layers=%d). Per CLAUDE.md rule #12, MTP "
                "is loaded for weight-key compat but NOT executed at Eagle3 "
                "inference. The Eagle3 draft head replaces it.",
                n_mtp,
                n_mtp,
            )
            # Placeholder modules so weight loader has somewhere to write.
            self.mtp = nn.ModuleList(
                [nn.Identity() for _ in range(n_mtp)]
            )
        else:
            self.mtp = nn.ModuleList()

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Functional RMSNorm matching V4 reference."""
        dtype = x.dtype
        x32 = x.float()
        var = x32.square().mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.norm_eps)
        return (self.final_norm_weight * x32).to(dtype)

    def _hc_head_fold(self, x: torch.Tensor) -> torch.Tensor:
        """Fold [b, s, hc_mult, d] -> [b, s, d] via the top-level mHC head.

        V4 reference ParallelHead.hc_head (lines 728-735). Differs from the
        per-Block hc_pre: this one uses scalar `hc_head_scale` (not the 3-vector
        scale used in Block.hc_pre) and only computes `pre` (no `post`/`comb`).
        """
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

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

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        positions: Optional[torch.Tensor] = None,
        return_aux_hidden_states: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """V4 forward with optional Eagle3 aux-hidden-state capture.

        Direct port of V4 reference Transformer.forward (lines 801-809) with
        Eagle3 aux capture grafted on per the qwen2.py:652 post-loop trap
        pattern.

        Args:
            input_ids: [B, T] token ids
            start_pos: token offset into the KV cache (0 = prefill)
            positions: unused (sglang interface compat); positions are derived
                from `start_pos` and seqlen internally
            return_aux_hidden_states: optional override of
                `self._enable_return_hidden_states`. When True, returns
                `(logits, aux_hidden_states_list)` instead of just `logits`.

        Returns:
            logits: [B, vocab_size] last-token logits (V4 reference takes
                only the last token via `x[:, -1]` in get_logits).
            (logits, aux_hs): when `return_aux_hidden_states` is True,
                aux_hs is a list of [B, T, hidden_size] tensors, one per
                layer index in `self._eagle3_layers_to_capture` (in the
                same order as the configured indices).

        Aux capture semantics:
            For each layer index i in self._eagle3_layers_to_capture, the
            aux tensor at that index is the layer's INPUT (i.e. the output
            of layer i-1, or the embed-expanded hidden state for i=0). We
            apply the qwen2.py:652 post-loop trap: when the configured
            list contains an index equal to num_hidden_layers (the
            "off-by-one for the final layer"), we capture the post-loop
            hidden state too. Both get folded over the hc_mult dim by mean
            so the per-layer aux feature dim is `hidden_size` (matching
            Eagle3 draft model expectations: 3 layers x 4096 dim = 12288
            aux feature dim for the V4-Flash slot triple [1, 21, 41]).
        """
        return_aux = (
            return_aux_hidden_states
            if return_aux_hidden_states is not None
            else self._enable_return_hidden_states
        )
        capture_layers = set(self._eagle3_layers_to_capture) if return_aux else set()

        # 1. Embed.
        h = self.embed(input_ids)
        # 2. Expand to hc_mult copies for Hyper-Connections (V4 ref line 805).
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        aux_hidden_states: List[torch.Tensor] = []
        end_layer = len(self.layers)

        # 3. Layer loop with aux capture INSIDE the loop (qwen2.py pattern).
        for i, layer in enumerate(self.layers):
            if i in capture_layers:
                # Capture the layer INPUT, mean over hc copies.
                # Shape: [B, T, hc_mult, d].mean(dim=-2) -> [B, T, d]
                aux_hidden_states.append(h.mean(dim=-2))
            h = layer(h, start_pos, input_ids)

        # 4. Post-loop trap: if end_layer (= num_hidden_layers) is in the
        # configured capture set, capture the final layer's OUTPUT.
        # set_eagle3_layers_to_capture sometimes maps indices [0, 14, 27]
        # to [1, 15, 28] (+1 offset to capture each layer's output as the
        # input to the next). For the last layer, that mapped index is
        # num_hidden_layers and is unreachable inside the for-loop.
        # Capture here, before the final norm + mHC fold, so the aux state
        # matches the same pre-norm convention as the in-loop captures.
        if end_layer in capture_layers:
            aux_hidden_states.append(h.mean(dim=-2))

        # 5. Top-level mHC head fold + final norm + lm_head.
        h_folded = self._hc_head_fold(h)  # [B, T, hc_mult, d] -> [B, T, d]
        h_normed = self._final_norm(h_folded)
        # V4 reference uses `x[:, -1]` to take only the last token's logits.
        # For sglang inference this matches the per-step generation pattern.
        logits = self.lm_head(h_normed[:, -1].float())

        if return_aux:
            return logits, aux_hidden_states
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load V4-Flash weights from the HF checkpoint into this model.

        TODO(phase1-loader): full body. The weight loader needs to:

        1. Map HF checkpoint keys to our parameter names. V4 uses
           `model.layers.<i>.{self_attn,mlp}.{wq_a,wq_b,wkv,wo_a,wo_b,...}`
           in its checkpoint; our names use a similar but not identical
           layout. Build the key map by inspecting `/tmp/v4-flash-meta/
           model.safetensors.index.json` and the V4 reference inference/
           convert.py.

        2. Handle FP4 expert weight dequantization. V4 stores expert weights
           in FP4 (`torch.float4_e2m1fn_x2`) with E8M0 scales. Our V4Expert
           uses nn.Linear (BF16 storage); the loader must dequantize FP4 -> BF16
           on load using the kernel.py:fp4_gemm logic. This is slow + memory-
           heavy at runtime but functionally correct. TODO(phase1-fp4) is to
           swap V4Expert to a V4-aware FP4 Linear that keeps weights packed.

        3. Handle FP8 non-expert weights similarly (e4m3fn with e8m0fnu scales).
           Sglang already has an FP8 weight loader path
           (sglang.srt.layers.quantization.fp8_kernel); reuse where possible.

        4. Handle the hash-routing tid2eid table (int32, no quant).

        5. Handle the MTP block weights — load them but don't run them
           (per CLAUDE.md rule #12). With `self.mtp = ModuleList(Identity)`
           the loader can write them to a flat parameter dict and they're
           ignored at forward time.

        Reference: deepseek_common/deepseek_weight_loader.py for the V3.2
        loader; the pattern carries over with FP4 modifications.
        """
        raise NotImplementedError(
            "DeepseekV4ForCausalLM.load_weights — see TODO(phase1-loader). "
            "Until this lands, instantiate the model with random weights "
            "for shape testing only."
        )


# ============================================================================
# Module-level export — explicit so the registry's pkgutil walk picks it up
# ============================================================================

EntryClass = DeepseekV4ForCausalLM
