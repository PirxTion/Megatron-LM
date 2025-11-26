import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
    )
    from megatron.core.post_training.modelopt.layers import Linear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear, TELinear, Linear = None, None, None
    HAVE_TE = False


@dataclass
class GLASelfAttentionSubmodules:
    """Submodules for the GLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_k_rope_proj: Union[ModuleSpec, type] = None  # Added for decoupled RoPE
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class GLAAttention(Attention):
    """Grouped Latent Attention layer abstract class.

    GLA extends MLA by using multiple latent heads (h_c) instead of a single one.
    Each latent head has dimension d_c = 2 * d_h (half of MLA's 4 * d_h).
    Query heads are split into groups, each group attending to one latent head.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: Union[GLASelfAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Number of latent heads for GLA (default to 2 if not specified)
        self.num_latent_heads = getattr(self.config, 'num_latent_heads', 2)
        
        # Group size: number of query heads per latent head
        assert self.config.num_attention_heads % self.num_latent_heads == 0, \
            f"num_attention_heads ({self.config.num_attention_heads}) must be divisible by num_latent_heads ({self.num_latent_heads})"
        self.group_size = self.config.num_attention_heads // self.num_latent_heads

        # Latent heads per partition for tensor parallelism
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert self.num_latent_heads % tp_size == 0 or tp_size % self.num_latent_heads == 0, \
            "num_latent_heads must be divisible by TP size or vice versa for proper sharding"
        
        if self.num_latent_heads >= tp_size:
            self.num_latent_heads_per_partition = self.num_latent_heads // tp_size
        else:
            # If TP > num_latent_heads, we can't shard latent heads cleanly without splitting dimensions.
            # For now, assume TP <= num_latent_heads or handle logic carefully.
            # The simplest GLA implementation assumes latent heads are the unit of sharding.
            raise ValueError(f"TP size ({tp_size}) cannot be larger than num_latent_heads ({self.num_latent_heads}) for GLA")

        # Calculate latent dimension per head
        # kv_lora_rank in config represents the TOTAL latent dimension across all heads
        assert self.config.kv_lora_rank % self.num_latent_heads == 0, \
            "Total latent dim (kv_lora_rank) must be divisible by num_latent_heads"
        self.latent_dim_per_head = self.config.kv_lora_rank // self.num_latent_heads

        # Overwrite the base class kv shape to support GLA
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == 'selective'
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

        # Output projection
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for grouped latent attention"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into GLA."
        assert attention_bias is None, "Attention bias should not be passed into GLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "GLA does not support Flash Decoding"

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None:
            raise NotImplementedError("GLA inference is not implemented.")

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
        )

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.attn_mask_type,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class GLASelfAttention(GLAAttention):
    """GLA Self-attention layer class"""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: GLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        # Q projection - same as MLA
        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_proj',
            )
        else:
            q_down_proj_kwargs = {}
            if submodules.linear_q_down_proj in [TELinear]:
                q_down_proj_kwargs['parallel_mode'] = 'duplicated'
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
            ]:
                q_down_proj_kwargs['gather_output'] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_down_proj',
                skip_weight_param_allocation=False,
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_up_proj',
            )

        # KV Down Projection (Latent)
        # Projects to the total latent dimension.
        # We use ColumnParallelLinear with gather_output=False to shard the latent vector across ranks.
        # Output shape per rank: [s, b, kv_lora_rank / TP]
        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs['parallel_mode'] = 'column'
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank, # Only project to latent dim
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_down_proj',
            skip_weight_param_allocation=False,
            **kv_down_proj_kwargs,
        )

        # KV RoPE Projection
        # Projects to the RoPE dimension.
        # We use ColumnParallelLinear with gather_output=True to ensure every rank gets the full RoPE vector.
        # This avoids duplicating the large latent vector while sharing the small RoPE vector.
        self.linear_kv_rope_proj = build_module(
            submodules.linear_k_rope_proj or submodules.linear_kv_down_proj, # Fallback to same type
            self.config.hidden_size,
            self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_rope_proj',
            gather_output=False, # Gather so all ranks have RoPE
        )

        # KV Up Projection
        # In GLA, each latent head projects to its specific group of query heads.
        # Since we have sharded the latent heads, we only project the local latent heads.
        # We use a ModuleList of local Linear layers to enforce the independence of latent heads.
        self.linear_kv_up_proj = nn.ModuleList()
        
        # Output dim for one latent head -> group of query heads
        up_proj_output_dim = self.group_size * (self.config.qk_head_dim + self.config.v_head_dim)
        
        kv_up_proj_kwargs = {}
        if submodules.linear_kv_up_proj in [TELinear]:
            kv_up_proj_kwargs['parallel_mode'] = 'duplicated'

        for _ in range(self.num_latent_heads_per_partition):
            # Local linear layer: latent_dim_per_head -> group_kv_dim
            # We use build_module to respect config (e.g. init method, dtype) but ensure it's local
            layer = build_module(
                submodules.linear_kv_up_proj,
                self.latent_dim_per_head,
                up_proj_output_dim,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='kv_up_proj',
                skip_weight_param_allocation=False,
                **kv_up_proj_kwargs,
            )
            self.linear_kv_up_proj.append(layer)

        # Layernorms
        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        # KV Layernorm
        # Applied per latent head. We initialize it with the dimension of a single latent head.
        # We use a ModuleList to ensure each latent head has its own learnable parameters.
        self.kv_layernorm = nn.ModuleList()
        for _ in range(self.num_latent_heads_per_partition):
            ln = build_module(
                submodules.kv_layernorm,
                hidden_size=self.latent_dim_per_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
            self.kv_layernorm.append(ln)

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, hidden_states, self.config, packed_seq_params
        )

        mscale = 1.0
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)

        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        # KV Down Projection (Latent) - Sharded output
        # kv_compressed: [s, b, kv_lora_rank / TP]
        kv_compressed, _ = self.linear_kv_down_proj(hidden_states)
        
        # KV RoPE Projection - Full output (gathered)
        # k_pos_emb: [s, b, qk_pos_emb_head_dim]
        k_pos_emb, _ = self.linear_kv_rope_proj(hidden_states)
        k_pos_emb = gather_from_tensor_model_parallel_region(k_pos_emb)

        if packed_seq_params is not None:
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            """
            # --- Query ---
            if self.config.q_lora_rank is not None:
                q_compressed = self.q_layernorm(q_compressed)
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                q, _ = self.linear_q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            # --- Key/Value (GLA Logic) ---
            
            # 1. Reshape to separate local latent heads
            # kv_compressed: [num_tokens, num_local_latents * latent_dim_per_head]
            # -> [num_tokens, num_local_latents, latent_dim_per_head]
            kv_reshaped = kv_compressed.view(*kv_compressed.size()[:-1], self.num_latent_heads_per_partition, self.latent_dim_per_head)
            
            # 2. Apply LayerNorm and Up-Projection per latent head
            # Each latent head projects to its group of query heads
            kv_outputs = []
            for i, (ln, layer) in enumerate(zip(self.kv_layernorm, self.linear_kv_up_proj)):
                # Select latent head i: [num_tokens, latent_dim_per_head]
                latent_head = kv_reshaped[..., i, :]
                
                # Apply LayerNorm
                normed_head = ln(latent_head)
                
                # Project: [num_tokens, group_size * (qk+v)]
                out, _ = layer(normed_head)
                kv_outputs.append(out)
            
            # 3. Concatenate outputs from all groups
            # kv: [num_tokens, num_local_latents * group_size * (qk+v)]
            # Note: num_local_latents * group_size = num_local_query_heads
            kv = torch.cat(kv_outputs, dim=-1)

            # kv: [num_tokens, n, (qk_head_dim + v_head_dim)]
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            q_len = q.size()[0]
            if packed_seq_params is None or self.config.context_parallel_size == 1:
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # q_no_pe: [num_tokens, n, qk_head_dim]
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_no_pe, q_pos_emb = torch.split(
                q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
            )

            # k_no_pe: [num_tokens, n, qk_head_dim]
            # value: [num_tokens, n, v_head_dim]
            k_no_pe, value = torch.split(
                kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
            )

            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )
            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )

            # query: [num_tokens, n, (qk_head_dim + qk_pos_emb_head_dim)]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            # key: [num_tokens, n, (qk_head_dim + qk_pos_emb_head_dim)]
            if k_pos_emb.ndim == 4:
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
            else:
                assert k_pos_emb.ndim == 3
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
            key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=self.config.fp8)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        return query, key, value
