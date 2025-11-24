# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import deprecate_inference_params


@dataclass
class GLASelfAttentionSubmodules:
    """Submodules for the GLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj_1: Union[ModuleSpec, type] = None
    linear_kv_up_proj_2: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv1_layernorm: Union[ModuleSpec, type] = None
    kv2_layernorm: Union[ModuleSpec, type] = None


class GLASelfAttention(Attention):
    """Training-only implementation of Gated Latent Attention.

    This layer follows the multi-latent attention structure but compresses the
    queries and keys into two latent spaces, applies RoPE on a dedicated
    positional sub-dimension, and pads the value tensor to match the Q/K
    dimensionality before computing attention. Inference is intentionally not
    implemented.
    """

    def __init__(
        self,
        config,
        submodules: GLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type="self",
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        self.head_dim = self.config.kv_channels
        self.rope_dim = getattr(self.config, "qk_pos_emb_head_dim", self.head_dim)
        self.q_proj_dim = 8 * self.head_dim
        self.kv_proj_dim = 4 * self.head_dim
        self.qk_dim = self.head_dim + self.rope_dim
        assert (
            self.num_attention_heads_per_partition % 2 == 0
        ), "GLA requires an even number of attention heads per tensor parallel rank."

        # Override attention dimensions to match the padded Q/K/V representation.
        self.key_hidden_size = self.qk_dim
        self.val_hidden_size = self.qk_dim

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.q_proj_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.kv1_layernorm = build_module(
            submodules.kv1_layernorm,
            hidden_size=self.kv_proj_dim // 2,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.kv2_layernorm = build_module(
            submodules.kv2_layernorm,
            hidden_size=self.kv_proj_dim // 2,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_q_proj = build_module(
            submodules.linear_q_proj,
            self.config.hidden_size,
            self.q_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='gla_q_down',
            tp_group=self.pg_collection.tp,
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.q_proj_dim,
            self.config.num_attention_heads * self.qk_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='gla_q_up',
            tp_group=self.pg_collection.tp,
        )

        self.linear_kv_proj = build_module(
            submodules.linear_kv_proj,
            self.config.hidden_size,
            self.kv_proj_dim + self.rope_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='gla_kv_down',
            tp_group=self.pg_collection.tp,
        )

        self.linear_kv_up_proj_1 = build_module(
            submodules.linear_kv_up_proj_1,
            self.kv_proj_dim // 2,
            self.config.num_attention_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='gla_kv_up_1',
            tp_group=self.pg_collection.tp,
        )
        self.linear_kv_up_proj_2 = build_module(
            submodules.linear_kv_up_proj_2,
            self.kv_proj_dim // 2,
            self.config.num_attention_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='gla_kv_up_2',
            tp_group=self.pg_collection.tp,
        )

        # Rebuild core attention with the padded Q/K/V dimensionality.
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.config.softmax_scale,
            k_channels=self.qk_dim,
            v_channels=self.qk_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

    def _maybe_gather(self, tensor: torch.Tensor, expected_last_dim: int) -> torch.Tensor:
        """Gather tensor model parallel shards when needed."""
        if tensor.size(-1) != expected_last_dim:
            tensor = gather_from_tensor_model_parallel_region(tensor)
            if self.config.sequence_parallel:
                tensor = scatter_to_sequence_parallel_region(tensor)
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context=None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        *,
        inference_params=None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None:
            raise NotImplementedError("GLA inference is not implemented.")
        assert key_value_states is None, "GLA only supports self-attention."
        assert rotary_pos_cos is None and rotary_pos_sin is None

        if self.config.no_rope_freq:
            no_rope = self.config.no_rope_freq[self.layer_number - 1]
            if no_rope:
                rotary_pos_emb = None

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'

        # =====================
        # Query projection path
        # =====================
        q_down, _ = self.linear_q_proj(hidden_states)
        q_down = self._maybe_gather(q_down, self.q_proj_dim)
        if packed_seq:
            q_down = q_down.squeeze(1)
        q_down = self.q_layernorm(q_down)

        q_up, _ = self.linear_q_up_proj(q_down)
        q_up = self._maybe_gather(q_up, self.config.num_attention_heads * self.qk_dim)
        if packed_seq:
            q_up = q_up.squeeze(1)
        q_up = q_up.view(
            *q_up.size()[:-1],
            self.num_attention_heads_per_partition,
            self.qk_dim,
        )

        q_content, q_rope = torch.split(q_up, [self.head_dim, self.rope_dim], dim=-1)

        # =====================
        # Key/Value projection
        # =====================
        kv_down, _ = self.linear_kv_proj(hidden_states)
        kv_down = self._maybe_gather(kv_down, self.kv_proj_dim + self.rope_dim)
        if packed_seq:
            kv_down = kv_down.squeeze(1)
        kv_content, k_rope = torch.split(kv_down, [self.kv_proj_dim, self.rope_dim], dim=-1)

        kv_first = kv_content[..., : self.kv_proj_dim // 2]
        kv_second = kv_content[..., self.kv_proj_dim // 2 :]
        kv_first = self.kv1_layernorm(kv_first)
        kv_second = self.kv2_layernorm(kv_second)

        kv_up_1, _ = self.linear_kv_up_proj_1(kv_first)
        kv_up_2, _ = self.linear_kv_up_proj_2(kv_second)

        kv_up_1 = self._maybe_gather(kv_up_1, self.config.num_attention_heads * self.head_dim)
        kv_up_2 = self._maybe_gather(kv_up_2, self.config.num_attention_heads * self.head_dim)

        if packed_seq:
            kv_up_1 = kv_up_1.squeeze(1)
            kv_up_2 = kv_up_2.squeeze(1)

        half_heads = self.num_attention_heads_per_partition // 2
        kv_up_1 = kv_up_1.view(*kv_up_1.size()[:-1], half_heads, 2 * self.head_dim)
        kv_up_2 = kv_up_2.view(*kv_up_2.size()[:-1], half_heads, 2 * self.head_dim)

        key_states = torch.zeros_like(q_up)
        value_states = torch.zeros_like(q_up)

        k1, v1 = torch.split(kv_up_1, self.head_dim, dim=-1)
        k2, v2 = torch.split(kv_up_2, self.head_dim, dim=-1)
        key_states[..., :half_heads, : self.head_dim] = k1
        value_states[..., :half_heads, : self.head_dim] = v1
        key_states[..., half_heads:, : self.head_dim] = k2
        value_states[..., half_heads:, : self.head_dim] = v2

        # =====================
        # Apply RoPE to Q/K
        # =====================
        k_rope = k_rope.unsqueeze(-2)

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

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

            if q_pos_emb is not None:
                q_rope = apply_rotary_pos_emb(
                    q_rope,
                    q_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    cp_group=self.pg_collection.cp,
                )
            if k_pos_emb is not None:
                k_rope = apply_rotary_pos_emb(
                    k_rope,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    cp_group=self.pg_collection.cp,
                )

        q_up = torch.cat([q_content, q_rope], dim=-1)
        key_states[..., self.head_dim :] = k_rope

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                q_up,
                key_states,
                value_states,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                q_up,
                key_states,
                value_states,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        core_attn_out = core_attn_out[..., : self.head_dim]

        if packed_seq:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        else:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), core_attn_out.size(1), -1)

        output, bias = self.linear_proj(core_attn_out)

        return output, bias
