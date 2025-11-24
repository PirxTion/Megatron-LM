# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.gla_attention import GLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils


def _get_gla_self_attn_submodules():
    return get_gpt_layer_local_spec(gla_attention=True).submodules.self_attention.submodules


class TestGLAAttention:
    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_forward_training_path(self):
        if not torch.cuda.is_available():
            pytest.skip("GLA unit test requires CUDA.")

        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            gla_attention=True,
            multi_latent_attention=False,
            use_cpu_initialization=True,
            qk_pos_emb_head_dim=8,
        )

        attention = GLASelfAttention(
            config,
            _get_gla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        seq_len, batch = 8, 2
        hidden_states = torch.randn(seq_len, batch, config.hidden_size, device='cuda')
        attention_mask = torch.ones((1, 1, seq_len, seq_len), device='cuda', dtype=torch.bool)

        output, bias = attention(hidden_states, attention_mask, rotary_pos_emb=None)

        assert output.shape == (seq_len, batch, config.hidden_size)
        assert bias.shape[0] == config.hidden_size

        with pytest.raises(NotImplementedError):
            attention(hidden_states, attention_mask, inference_context=object())
