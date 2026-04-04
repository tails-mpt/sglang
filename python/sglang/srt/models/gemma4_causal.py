# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
SGLang model implementation for Gemma-4 (text-only / causal LM).

Based on the Gemma-3 SGLang implementation with the following Gemma-4 additions:
  - text_config extraction from nested HuggingFace config
  - model.language_model.* weight prefix stripping
  - per-layer learned scalar (layer_scalar)
  - attention_k_eq_v support (shared k/v projections)
  - EAGLE-3 speculative decoding hooks

NOTE: Does NOT import any Gemma4-specific classes from transformers (not available
in transformers<=4.57.1). Config is read via getattr on PretrainedConfig.
"""

import copy
from typing import Iterable, List, Optional, Set, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    ROPE_INIT_FUNCTIONS,
    PretrainedConfig,
    PreTrainedModel,
)

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.layernorm import Gemma3RMSNorm


class Gemma4RMSNorm(nn.Module):
    """Gemma-4 RMSNorm: uses weight directly (not 1+weight like Gemma-3)."""
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(x)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, make_layers


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


def extract_layer_index(prefix: str) -> int:
    """Extract the layer index from a prefix string."""
    parts = prefix.split(".")
    for part in parts:
        if part.startswith("layers."):
            layer_str = part.split(".")[-1]
            try:
                return int(layer_str)
            except ValueError:
                continue
    return -1


class Gemma4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma4 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        hidden_size = config.hidden_size

        # Gemma-4 uses different head_dim for sliding vs full attention layers
        layer_types = getattr(config, "layer_types", [])
        is_sliding = layer_types[layer_id] == "sliding_attention" if layer_id < len(layer_types) else True
        self.is_sliding = is_sliding

        if is_sliding:
            head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
            num_kv_heads_for_layer = self.total_num_kv_heads
        else:
            # Full attention layers use global_head_dim and may have fewer KV heads
            head_dim = getattr(config, "global_head_dim", getattr(config, "head_dim", hidden_size // config.num_attention_heads))
            num_kv_heads_for_layer = getattr(config, "num_global_key_value_heads", self.total_num_kv_heads)

        self.head_dim = head_dim
        self.total_num_kv_heads_layer = num_kv_heads_for_layer
        self.num_kv_heads = max(1, num_kv_heads_for_layer // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = getattr(config, "query_pre_attn_scalar", head_dim)**-0.5

        self.attention_k_eq_v = getattr(config, "attention_k_eq_v", False)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            num_kv_heads_for_layer,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Sliding window config (RoPE is handled by Gemma4TextModel's rotary embeddings)
        if self.is_sliding:
            self.sliding_window = get_attention_sliding_window_size(config)
        else:
            self.sliding_window = None

        # With SWAKVPool, each layer type has its own KV cache with native dimensions.
        # No padding/repeating needed — use layer-native dims directly.
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        # Gemma-4 adds normalization for q, k, AND v
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        # v_norm: RMSNorm WITHOUT learnable scale (no parameters, just normalization)
        self.v_norm_eps = config.rms_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Gemma-4: V comes from v_proj (via qkv_proj split), NOT from K clone
        # attention_k_eq_v doesn't mean shared projections — checkpoint has distinct v_proj weights

        # [s, h, head_dim]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        q = self.q_norm(q)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = k.transpose(0, 1).unsqueeze(0)
        k = self.k_norm(k)

        # Gemma-4: v_norm is RMSNorm without scale
        # Temporarily disabled for debugging — checking if v_proj alone is correct
        # v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        # v_float = v.float()
        # v = (v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + self.v_norm_eps)).to(v.dtype)
        # v = v.flatten(-2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # [b, h, s, head_dim] ->  [b, s, h, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        # With SWAKVPool, each layer type has its own KV cache — no padding needed

        # DEBUG: print shapes for first forward pass
        if self.layer_id == 0 and not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            import sys
            print(f"[DEBUG L{self.layer_id} {'sliding' if self.is_sliding else 'global'}] "
                  f"q={q.shape} k={k.shape} v_flat={v.shape} "
                  f"head_dim={self.head_dim} num_kv={self.num_kv_heads}", file=sys.stderr, flush=True)
        if self.layer_id == 5 and not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            import sys
            print(f"[DEBUG L{self.layer_id} {'sliding' if self.is_sliding else 'global'}] "
                  f"q={q.shape} k={k.shape} v_flat={v.shape} "
                  f"head_dim={self.head_dim} num_kv={self.num_kv_heads}", file=sys.stderr, flush=True)

        attn_output = self.attn(q, k, v, forward_batch=forward_batch)

        # Handle output shape (triton backend returns [1, s, h, head_dim])
        if attn_output.dim() == 4 and attn_output.shape[0] == 1:
            attn_output = attn_output.squeeze(0)
            attn_output = attn_output.flatten(-2, -1)

        output, _ = self.o_proj(attn_output)
        return output


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4Attention(
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding
        self.layer_id = layer_id

        # Gemma-4: per-layer learned scalar that multiplies the layer residual contribution
        # Initialized to 1.0; actual value loaded from weights
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            forward_batch=forward_batch,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Gemma-4: layer_scalar scales the ENTIRE layer output (not just the MLP contribution)
        hidden_states = hidden_states * self.layer_scalar

        outputs = (hidden_states,)

        return outputs


class Gemma4RotaryEmbedding(nn.Module):
    """
    Gemma-4 rotary embedding supporting both sliding (default) and full (proportional) attention.

    Following HF's Gemma4TextRotaryEmbedding:
    - Sliding layers: standard RoPE with head_dim=256, theta=10000
    - Full layers: proportional RoPE with global_head_dim=512, theta=1M, partial_rotary_factor=0.25
      The proportional init produces inv_freq with dim=256 (64 rotary + 192 zero-freq),
      yielding cos/sin of dim=512 that match global_head_dim.
    """
    def __init__(self, config: PretrainedConfig, device=None, layer_type="sliding_attention"):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_type = layer_type

        # Extract per-layer-type RoPE parameters from Gemma-4's nested config
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        layer_params = rope_params.get(layer_type, {}) if isinstance(rope_params, dict) and layer_type in rope_params else rope_params

        rope_type = layer_params.get("rope_type", "default") if isinstance(layer_params, dict) else "default"
        self.rope_type = rope_type

        if rope_type == "default" or rope_type not in ROPE_INIT_FUNCTIONS:
            # Standard RoPE: compute inv_freq directly
            base = layer_params.get("rope_theta", 10000.0) if isinstance(layer_params, dict) else 10000.0
            dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            self.attention_scaling = 1.0
        elif rope_type == "proportional":
            # Proportional RoPE for full attention layers
            # Must pass head_dim_key="global_head_dim" so it uses 512 instead of 256
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(
                config, device=device, layer_type=layer_type, head_dim_key="global_head_dim"
            )
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(config, device=device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: Optional[float] = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class Gemma4TextModel(PreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )

        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Gemma-4 uses per-layer-type RoPE:
        # - Global (full_attention): proportional RoPE, global_head_dim=512, partial_rotary_factor=0.25
        # - Local (sliding_attention): standard RoPE, head_dim=256
        # Following HF's Gemma4TextRotaryEmbedding approach
        self.rotary_emb = Gemma4RotaryEmbedding(config=config, layer_type="full_attention")
        self.rotary_emb_local = Gemma4RotaryEmbedding(config=config, layer_type="sliding_attention")

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma4DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # For EAGLE-3 support
        self.layers_to_capture = []

        # Skip post_init() — SGLang handles weight loading via load_weights()
        # self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        # DEBUG: track hidden state magnitude through layers
        if not hasattr(self, '_layer_debug_done'):
            self._layer_debug_done = True
            self._debug_mode = True
            import sys
            h = hidden_states.float()
            print(f"[DEBUG] Embedding: mean={h.mean():.4f} std={h.std():.4f} "
                  f"min={h.min():.4f} max={h.max():.4f}", file=sys.stderr, flush=True)
        else:
            self._debug_mode = False

        if positions.dim() == 1:
            positions = einops.rearrange(positions, "s -> 1 s")

        position_embeddings_global = self.rotary_emb(hidden_states, positions)
        position_embeddings_local = self.rotary_emb_local(hidden_states, positions)

        aux_hidden_states = []
        for i, layer in enumerate(self.layers):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)
            layer_outputs = layer(
                positions=positions,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if getattr(self, '_debug_mode', False) and i in [0, 5, 10, 30, 59]:
                import sys
                h = hidden_states.float()
                print(f"[DEBUG] After layer {i}: mean={h.mean():.4f} std={h.std():.4f} "
                      f"min={h.min():.4f} max={h.max():.4f}", file=sys.stderr, flush=True)

        hidden_states = self.norm(hidden_states)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Gemma4ForCausalLM(PreTrainedModel):
    # transformers 5.5.0 expects dict for tied weights; use empty dict since we handle tying manually
    _tied_weights_keys = {}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    base_model_prefix = "language_model"

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Gemma-4 HF config nests text model config under text_config.
        # Our transformers version doesn't have Gemma4TextConfig, so we
        # extract it manually from the PretrainedConfig object.
        text_config = getattr(config, "text_config", config)

        # If text_config came as a dict (some transformers versions), convert to namespace
        if isinstance(text_config, dict):
            from types import SimpleNamespace

            text_config = SimpleNamespace(**text_config)

        super().__init__(config=text_config)
        self.config = text_config
        self.quant_config = quant_config
        self.model = Gemma4TextModel(
            text_config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(text_config)

        if getattr(text_config, "tie_word_embeddings", True):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        # For EAGLE-3 support
        self.capture_aux_hidden_states = False

        # Skip post_init() — SGLang handles weight loading via load_weights().
        # post_init() triggers _init_weights which calls ROPE_INIT_FUNCTIONS on
        # our Gemma4RotaryEmbedding without the required layer_type context.
        # self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ---- EAGLE-3 hooks ----

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if embed.dtype != torch.bfloat16:
            embed = embed.to(torch.bfloat16)
        if head.dtype != torch.bfloat16:
            head = head.to(torch.bfloat16)
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        if embed.dtype != torch.bfloat16:
            embed = embed.to(torch.bfloat16)
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            # +1 because sglang captures the INPUT to the layer, so layer i
            # captures the output of layer i-1
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    # ---- Forward ----

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> LogitsProcessor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, **kwargs
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        result = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

        # DEBUG: print first forward pass logits
        if not hasattr(self, '_logits_debug_printed'):
            self._logits_debug_printed = True
            import sys
            if hasattr(result, 'next_token_logits') and result.next_token_logits is not None:
                logits = result.next_token_logits
                top5 = logits[0].topk(5)
                print(f"[DEBUG LOGITS] shape={logits.shape}, top5_vals={top5.values.tolist()}, top5_ids={top5.indices.tolist()}", file=sys.stderr, flush=True)
            print(f"[DEBUG HIDDEN] hidden_states shape={hidden_states.shape}, "
                  f"mean={hidden_states.float().mean():.6f}, std={hidden_states.float().std():.6f}", file=sys.stderr, flush=True)

        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            # Strip the multimodal wrapper prefix: model.language_model.* -> model.*
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]

            # Gemma-4 attention_k_eq_v: v_proj weights are the same as k_proj.
            # When loading, if we encounter v_proj, load it as k_proj duplicate
            # into the qkv_proj shard for v.
            # (The QKVParallelLinear already allocates separate k and v shards.)

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip lm_head.weight when tied with embed_tokens
                if "lm_head.weight" in name:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # DEBUG: verify layernorm weights loaded correctly
        import sys
        all_params = set(params_dict.keys())
        missing = all_params - loaded_params
        if missing:
            print(f"[WARN] {len(missing)} params not loaded:", file=sys.stderr, flush=True)
            for m in sorted(missing)[:10]:
                print(f"  {m}: {params_dict[m].shape}", file=sys.stderr, flush=True)

        # Check specific weights
        for key in ["model.layers.0.input_layernorm.weight", "model.norm.weight"]:
            if key in params_dict:
                w = params_dict[key]
                print(f"[DEBUG WEIGHT] {key}: mean={w.float().mean():.4f} std={w.float().std():.4f}", file=sys.stderr, flush=True)

        return loaded_params


class Gemma4ForConditionalGeneration(Gemma4ForCausalLM):
    """Alias for the conditional generation architecture name used by HuggingFace."""

    pass


EntryClass = [Gemma4ForCausalLM, Gemma4ForConditionalGeneration]
