# Copyright 2025-2026 SGLang Team
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
Inference-only GLM-4.7-Flash (GLM-4 MoE Lite) model with Multi-head Latent Attention (MLA).

GLM-4.7-Flash uses MLA architecture (similar to DeepSeek V2) with:
- LoRA-compressed query projections (q_a_proj -> q_a_layernorm -> q_b_proj)
- LoRA-compressed KV projections (kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj)
- Interleaved RoPE (non-neox style)
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, make_layers

# Import shared components from glm4_moe
from sglang.srt.models.glm4_moe import (
    Glm4MoeMLP,
    Glm4MoeGate,
    Glm4MoeSparseMoeBlock,
)

_is_cuda = is_cuda()
logger = logging.getLogger(__name__)


class Glm4MoeLiteAttentionMLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) for GLM-4.7-Flash.

    MLA uses LoRA-style compression for queries and key-values:
    - Query: hidden -> q_a_proj -> layernorm -> q_b_proj -> Q
    - KV: hidden -> kv_a_proj_with_mqa -> (split) -> layernorm -> kv_b_proj -> K, V

    The key insight is that K is split into:
    - k_nope: non-positional embedding part (from kv_b_proj)
    - k_rope: positional embedding part (directly from kv_a_proj_with_mqa)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        # MLA config parameters
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        rms_norm_eps = config.rms_norm_eps

        # Query path: hidden -> q_lora_rank -> num_heads * qk_head_dim
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_a_proj", prefix),
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        # KV path: hidden -> kv_lora_rank + qk_rope_head_dim -> num_heads * (qk_nope_head_dim + v_head_dim)
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        # RoPE - GLM uses interleaved style (is_neox_style=False)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        # Attention
        self.attn = RadixAttention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,  # MLA effectively has same Q and KV heads after expansion
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Query path
        q_compressed, _ = self.q_a_proj(hidden_states)
        q_compressed = self.q_a_layernorm(q_compressed)
        q, _ = self.q_b_proj(q_compressed)

        # KV path
        kv_compressed, _ = self.kv_a_proj_with_mqa(hidden_states)
        # Split: first kv_lora_rank dims go through layernorm, last qk_rope_head_dim are k_rope
        kv_compressed_for_norm = kv_compressed[..., :self.kv_lora_rank]
        k_rope = kv_compressed[..., self.kv_lora_rank:]

        kv_compressed_normed = self.kv_a_layernorm(kv_compressed_for_norm)
        kv, _ = self.kv_b_proj(kv_compressed_normed)

        # Reshape q: (batch*seq, num_local_heads * qk_head_dim) -> (batch*seq, num_local_heads, qk_head_dim)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.contiguous()
        q_rope = q_rope.contiguous()

        # Reshape kv: split into k_nope and v
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.contiguous()
        v = v.contiguous()

        # k_rope is shared across heads (single head from MQA), keep as (tokens, 1, rope_dim)
        k_rope = k_rope.view(-1, 1, self.qk_rope_head_dim).contiguous()

        # Apply RoPE to rope portions
        q_rope_for_rope = q_rope.view(-1, self.num_local_heads * self.qk_rope_head_dim)
        k_rope_for_rope = k_rope.view(-1, self.qk_rope_head_dim)

        q_rope_rotated, k_rope_rotated = self.rotary_emb(positions, q_rope_for_rope, k_rope_for_rope)

        # Reshape back
        q_rope = q_rope_rotated.view(-1, self.num_local_heads, self.qk_rope_head_dim)
        k_rope = k_rope_rotated.view(-1, 1, self.qk_rope_head_dim).expand(-1, self.num_local_heads, -1).contiguous()

        # Concatenate nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Flatten for attention - shape: (tokens, num_heads * head_dim)
        q = q.view(-1, self.num_local_heads * self.qk_head_dim)
        k = k.view(-1, self.num_local_heads * self.qk_head_dim)
        v = v.view(-1, self.num_local_heads * self.v_head_dim)

        # Attention
        attn_output = self.attn(q, k, v, forward_batch)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class Glm4MoeLiteDecoderLayer(nn.Module):
    """Decoder layer for GLM-4.7-Flash using MLA attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.layer_id = layer_id

        # Always use MLA attention for GLM-4.7-Flash
        self.self_attn = Glm4MoeLiteAttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        # MLP: sparse (MoE) or dense based on layer type
        self.is_layer_sparse = self._is_layer_sparse(layer_id)

        if self.is_layer_sparse:
            self.mlp = Glm4MoeSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
            )
        else:
            self.mlp = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _is_layer_sparse(self, layer_id: int) -> bool:
        return (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input layernorm + residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Post-attention layernorm + residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states, forward_batch)

        return hidden_states, residual


class Glm4MoeLiteModel(nn.Module):
    """GLM-4.7-Flash base model."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Glm4MoeLiteDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture = []

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Glm4MoeLiteForCausalLM(nn.Module):
    """GLM-4.7-Flash for causal language modeling with MLA attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.num_fused_shared_experts = self._determine_num_fused_shared_experts()

        self.model = Glm4MoeLiteModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def _determine_num_fused_shared_experts(self) -> int:
        if get_global_server_args().disable_shared_experts_fusion:
            return 0
        if not getattr(self.config, "n_shared_experts", None):
            return 0
        if not _is_cuda:
            return 0
        return self.config.n_shared_experts

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # MLA models don't use QKV stacking - weights load directly
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{self.config.n_routed_experts}",
                )

            if "rotary_emb.inv_freq" in name:
                continue

            # Handle stacked params (gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle expert weights
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        continue

                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = [Glm4MoeLiteForCausalLM]
