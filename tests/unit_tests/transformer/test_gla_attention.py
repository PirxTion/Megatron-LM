import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.gla_attention import GLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

# ==========================================
# Reference Implementation (Modified to use torch SDPA)
# ==========================================

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states.to(input_dtype))

class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings


        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # If dim is provided, we need to ensure the config passed to rope_init_fn reflects it.
        # This is necessary because transformers' rope_init_fn typically uses config.head_dim.
        init_config = self.config
        if dim is not None and self.config is not None:
            import copy
            init_config = copy.copy(self.config)
            init_config.head_dim = dim

        inv_freq, self.attention_scaling = self.rope_init_fn(init_config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rope_dim = self.config.qk_rope_dim 
        self.rotary_emb = LlamaRotaryEmbedding(dim=self.rope_dim, config=self.config) 

class LlamaFlashGLA(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.kv_proj_dim = 4 * self.head_dim   #2 latent heads each 2*d 
        self.q_proj_dim = 8 * self.head_dim

        self.q_norm = LlamaRMSNorm(self.q_proj_dim, self.config.rms_norm_eps)
        self.kv_norm_1 = LlamaRMSNorm(self.kv_proj_dim // 2, self.config.rms_norm_eps)
        self.kv_norm_2 = LlamaRMSNorm(self.kv_proj_dim // 2, self.config.rms_norm_eps)

        #Query
        self.W_dQ = nn.Linear(self.hidden_size, self.q_proj_dim)         
        self.W_uQ_rope = nn.Linear(self.q_proj_dim, self.num_heads * (self.head_dim + self.rope_dim)) 

        self.W_dKV = nn.Linear(self.hidden_size, self.kv_proj_dim + self.rope_dim) 
        
        self.W_ukv_1 = nn.Linear(self.kv_proj_dim // 2, (self.num_heads * self.head_dim)) 
        self.W_ukv_2 = nn.Linear(self.kv_proj_dim // 2, (self.num_heads * self.head_dim)) 

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        q_proj = self.W_dQ(hidden_states)

        query_states = self.q_norm(q_proj)

        query_states = self.W_uQ_rope(query_states).view(bsz, q_len, self.num_heads, self.head_dim + self.rope_dim)

        query_rope = query_states[..., self.head_dim:]

        KV_compressed_cache = self.W_dKV(hidden_states)

        compressed_kv, key_rope = torch.split(KV_compressed_cache, [self.kv_proj_dim, self.rope_dim], dim=-1)
        key_rope = key_rope.view(bsz, q_len, 1, self.rope_dim)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(query_rope, position_ids) 
        else:
            cos, sin = position_embeddings

        key_states = torch.zeros_like(query_states)  
        value_states = torch.zeros_like(query_states) 
        
        latent_dim_per_head = self.kv_proj_dim // 2        
        half_heads  = self.num_heads // 2         

        # c_1= 2*d per head
        kv_first_half   = compressed_kv[..., :latent_dim_per_head]          
        kv_first_half   = self.kv_norm_1(kv_first_half)             

        KV_compressed_1 = self.W_ukv_1(kv_first_half)               

        KV_compressed_1 = KV_compressed_1.view(bsz, q_len, half_heads, 2 * self.head_dim)

        key_states[:, :, :half_heads, :self.head_dim]   = KV_compressed_1[:, :, :, :self.head_dim]   
        value_states[:, :, :half_heads, :self.head_dim] = KV_compressed_1[:, :, :,  self.head_dim:]  

        # c_2 = 2*d per head
        kv_second_half  = compressed_kv[..., latent_dim_per_head:]          
        kv_second_half  = self.kv_norm_2(kv_second_half)           


        KV_compressed_2 = self.W_ukv_2(kv_second_half)              

        KV_compressed_2 = KV_compressed_2.view(bsz, q_len, half_heads, 2 * self.head_dim)

        key_states[:, :, half_heads:, :self.head_dim]   = KV_compressed_2[:, :, :, :self.head_dim]   
        value_states[:, :, half_heads:, :self.head_dim] = KV_compressed_2[:, :, :,  self.head_dim:]  

        query_rope, key_rope = apply_rotary_pos_emb(query_rope, key_rope, cos, sin, unsqueeze_dim=2)


        query_states[..., self.head_dim:].copy_(query_rope)
        key_states[..., self.head_dim:].copy_(key_rope)

        # Replaced flash_attn_func with torch.nn.functional.scaled_dot_product_attention
        # Input shape: (B, S, H, D) -> Transpose to (B, H, S, D) for SDPA
        q = query_states.transpose(1, 2)


        k = key_states.transpose(1, 2)


        v = value_states.transpose(1, 2)


        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )

        # Transpose back to (B, S, H, D)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output[..., :self.head_dim].contiguous() 
        attn_output = attn_output.view(bsz, q_len, -1) 
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

# ==========================================
# End Reference Implementation
# ==========================================


def _get_gla_self_attn_submodules():
    return get_gpt_layer_with_transformer_engine_spec(gla_attention=True).submodules.self_attention.submodules


class TestGLAAttention:
    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        # torch.manual_seed(123)
        # torch.cuda.manual_seed(123)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_constructor(self):
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gla_attention=True,
            multi_latent_attention=False,
            use_cpu_initialization=True,
            qk_pos_emb_head_dim=16,
            kv_lora_rank=64,
            q_lora_rank=64,
            num_latent_heads=2,
        )
        attention = GLASelfAttention(
            config,
            _get_gla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        assert isinstance(attention, GLASelfAttention)
        assert len(attention.linear_kv_up_proj) == 2
        assert len(attention.kv_layernorm) == 2

    def test_forward_training_path(self):
        if not torch.cuda.is_available():
            pytest.skip("GLA unit test requires CUDA.")

        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gla_attention=True,
            multi_latent_attention=False,
            use_cpu_initialization=True,
            qk_pos_emb_head_dim=16,
            kv_lora_rank=64,
            q_lora_rank=64,
            qk_head_dim=32,
            v_head_dim=32,
            num_latent_heads=2,
        )

        attention = GLASelfAttention(
            config,
            _get_gla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        seq_len, batch = 32, 2
        hidden_states = torch.randn(seq_len, batch, config.hidden_size, device='cuda')
        attention_mask = torch.ones((1, 1, seq_len, seq_len), device='cuda', dtype=torch.bool)

        output, bias = attention(hidden_states, attention_mask, rotary_pos_emb=None)

        assert output.shape == (seq_len, batch, config.hidden_size)
        assert bias.shape[0] == config.hidden_size

    def test_correctness_vs_reference(self):
        if not torch.cuda.is_available():
            pytest.skip("GLA unit test requires CUDA.")
        
        # 1. Configuration
        hidden_size = 128
        num_heads = 4
        head_dim = 32
        rope_dim = 16
        
        # Megatron Config
        megatron_config = MLATransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            gla_attention=True,
            multi_latent_attention=False,
            use_cpu_initialization=True,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            qk_pos_emb_head_dim=rope_dim,
            kv_lora_rank=4 * head_dim, # Match Ref kv_proj_dim
            q_lora_rank=8 * head_dim,  # Match Ref q_proj_dim
            num_latent_heads=2,
            rotary_base=10000,
            layernorm_epsilon=1e-6,
            add_bias_linear=False,
            rope_type="rope",
        )

        # Reference Config
        ref_config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=head_dim,
            qk_rope_dim=rope_dim,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            _attn_implementation="sdpa",
            max_position_embeddings=2048,
        )

        # 2. Instantiate Models
        megatron_model = GLASelfAttention(
            megatron_config,
            _get_gla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        ref_model = LlamaFlashGLA(ref_config, layer_idx=0).cuda()

        # 3. Copy Weights (Ref -> Megatron)
        with torch.no_grad():
            # --- Query Down ---
            megatron_model.linear_q_down_proj.weight.copy_(ref_model.W_dQ.weight)
            ref_model.W_dQ.bias.zero_()

            # --- Query Norm ---
            megatron_model.q_layernorm.weight.copy_(ref_model.q_norm.weight)

            # --- Query Up + RoPE ---
            megatron_model.linear_q_up_proj.weight.copy_(ref_model.W_uQ_rope.weight)
            ref_model.W_uQ_rope.bias.zero_()

            # --- KV Down (Latent + RoPE) ---
            w_dkv = ref_model.W_dKV.weight # [kv_dim + rope_dim, hidden]
            kv_dim = megatron_config.kv_lora_rank
            rope_dim = megatron_config.qk_pos_emb_head_dim
            
            w_latent, w_rope = torch.split(w_dkv, [kv_dim, rope_dim], dim=0)
            
            megatron_model.linear_kv_down_proj.weight.copy_(w_latent)
            megatron_model.linear_kv_rope_proj.weight.copy_(w_rope)
            ref_model.W_dKV.bias.zero_()

            # --- KV Norms ---
            megatron_model.kv_layernorm[0].weight.copy_(ref_model.kv_norm_1.weight)
            megatron_model.kv_layernorm[1].weight.copy_(ref_model.kv_norm_2.weight)

            # --- KV Up ---
            megatron_model.linear_kv_up_proj[0].weight.copy_(ref_model.W_ukv_1.weight)
            megatron_model.linear_kv_up_proj[1].weight.copy_(ref_model.W_ukv_2.weight)
            ref_model.W_ukv_1.bias.zero_()
            ref_model.W_ukv_2.bias.zero_()

            # --- Output ---
            megatron_model.linear_proj.weight.copy_(ref_model.o_proj.weight)

        # 4. Run Forward
        seq_len = 16
        batch_size = 2
        hidden_states = torch.randn(seq_len, batch_size, hidden_size, device='cuda')
        
        # Set to eval mode to disable dropout
        megatron_model.eval()
        ref_model.eval()

        # Ref expects (B, S, H)
        hidden_states_ref = hidden_states.permute(1, 0, 2).clone()
        
        # Ref
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        out_ref, _, _ = ref_model(hidden_states_ref, position_ids=position_ids)

        # Megatron
        out_megatron, _ = megatron_model(hidden_states, attention_mask=None)
        
        # Compare
        out_ref_sbh = out_ref.permute(1, 0, 2)

        print("Output Megatron:", out_megatron)
        print("Output Reference:", out_ref_sbh)
        
        # Check tolerances
        # BF16/FP32 differences might exist if not careful, but we are using float32 (default) here?
        # Torch default is float32.
        
        # Note: RoPE implementation details (interleaved vs half) might differ.
        # Megatron `apply_rotary_pos_emb` usually does `rotate_half` ([-x2, x1]).
        # Ref `rotate_half` does ([-x2, x1]).
        # They seem consistent.
        

        torch.testing.assert_close(out_megatron, out_ref_sbh, rtol=1e-3, atol=1e-3)
