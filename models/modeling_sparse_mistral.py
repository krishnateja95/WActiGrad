
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from prune_utils import *
from prune_utils_FFN import PruneGPT_FFN_hidden_size
from prune_utils_attention import PruneGPT_Attention

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.mistral.configuration_mistral import MistralConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
    def prune_weight(self, prune_mask):
        prune_mask = prune_mask.to(device = self.weight.device)
        self.weight = nn.Parameter((self.weight[prune_mask]).to(device = self.weight.device))
        del prune_mask


class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size,       self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size,       self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,       bias=False)
        self.act_fn    = ACT2FN[config.hidden_act]

    def initialize_FFN_pruning_functions(self):
        self.prune_API_MLP_FFN_1      =  PruneGPT_FFN_hidden_size(prune_dimension = self.intermediate_size)
        self.prune_API_MLP_FFN_2      =  PruneGPT_FFN_hidden_size(prune_dimension = self.intermediate_size)
        self.prune_API_MLP_attention  =  PruneGPT_FFN_hidden_size(prune_dimension = self.intermediate_size)
        self.prune_API_MLP_FFN_3      =  PruneGPT_FFN_hidden_size(prune_dimension = self.hidden_size)
        
    def save_FFN_device(self):
        self.prune_API_MLP_FFN_1.save_FFN_device(self.gate_proj.weight.device)
        self.prune_API_MLP_FFN_2.save_FFN_device(self.gate_proj.weight.device)
        self.prune_API_MLP_attention.save_FFN_device(self.gate_proj.weight.device)
        self.prune_API_MLP_FFN_3.save_FFN_device(self.gate_proj.weight.device)

    def store_FFN_gradients(self):
        self.prune_API_MLP_FFN_1.store_FFN_gradients(self.gate_proj.weight.grad)
        self.prune_API_MLP_FFN_2.store_FFN_gradients(self.up_proj.weight.grad)
        self.prune_API_MLP_FFN_3.store_FFN_gradients(self.down_proj.weight.grad)
        

    def store_FFN_weight_gradients(self):
        self.prune_API_MLP_FFN_1.store_FFN_weight_gradients(self.gate_proj.weight, self.gate_proj.weight.grad)
        self.prune_API_MLP_FFN_2.store_FFN_weight_gradients(self.up_proj.weight,   self.up_proj.weight.grad)
        self.prune_API_MLP_FFN_3.store_FFN_weight_gradients(self.down_proj.weight, self.down_proj.weight.grad)


    def update_FFN_activation_samples(self):
        self.prune_API_MLP_FFN_1.update_FFN_activation_samples()
        self.prune_API_MLP_FFN_2.update_FFN_activation_samples()
        self.prune_API_MLP_FFN_3.update_FFN_activation_samples()
        self.prune_API_MLP_attention.update_FFN_activation_samples()
        
    def set_FFN_calibration(self, calibration):
        self.calibration = calibration
        
    def set_FFN_pruning_choices(self, args):
        self.prune_API_MLP_FFN_1.set_FFN_pruning_choices(args)
        self.prune_API_MLP_FFN_2.set_FFN_pruning_choices(args)
        self.prune_API_MLP_FFN_3.set_FFN_pruning_choices(args)
        self.prune_API_MLP_attention.set_FFN_pruning_choices(args)
        
        self.prune_API_MLP_FFN_3.FFN_act_score = False

        self.prune_MLP          = args.prune_MLP
        self.prune_MLP_sparsity = args.prune_MLP_sparsity
        self.prune_hidden_size  = args.prune_hidden_size

        self.FFN_wt_score    =  args.FFN_wt_score
        self.FFN_act_score   =  args.FFN_act_score
        self.FFN_grad_score  =  args.FFN_grad_score

        self.wt_plus_grad_plus_act = args.wt_plus_grad_plus_act
        self.wt_mult_grad_plus_act = args.wt_mult_grad_plus_act

    def store_FFN_weights(self):
        self.prune_API_MLP_FFN_1.store_FFN_weights(self.gate_proj.weight)
        self.prune_API_MLP_FFN_2.store_FFN_weights(self.up_proj.weight)
        self.prune_API_MLP_FFN_3.store_FFN_weights(self.down_proj.weight)

    
    def get_FFN_hidden_size_score(self):
        hidden_size_score = 0
        if self.prune_hidden_size:
            hidden_size_score = self.prune_API_MLP_FFN_3.get_FFN_layer_score() 
        return hidden_size_score



    def prune_FFN_fc_layer(self, global_hidden_size_prune_mask):

        if self.prune_hidden_size:
            self.gate_proj     = prune_input_linear_layer(self.gate_proj,  global_hidden_size_prune_mask)
            self.up_proj       = prune_input_linear_layer(self.up_proj,    global_hidden_size_prune_mask)
            self.down_proj     = prune_output_linear_layer(self.down_proj, global_hidden_size_prune_mask)
            
            
        if self.prune_MLP:
            MLP_score  = self.prune_API_MLP_FFN_1.get_FFN_layer_score()
            MLP_score += self.prune_API_MLP_FFN_2.get_FFN_layer_score()
            
            prune_mask_FFN_1_2  = get_prune_mask_from_score(prune_dimension = self.intermediate_size,
                                                            score = MLP_score,
                                                            sparsity = self.prune_MLP_sparsity,
                                                            prune_dim_nearest = 16)
            
            self.gate_proj     = prune_output_linear_layer(self.gate_proj, prune_mask_FFN_1_2)
            self.up_proj       = prune_output_linear_layer(self.up_proj,   prune_mask_FFN_1_2)
            self.down_proj     = prune_input_linear_layer(self.down_proj,  prune_mask_FFN_1_2)
    

    def forward(self, x):
        y = self.act_fn(self.gate_proj(x))
        if self.calibration and self.FFN_act_score:
            self.prune_API_MLP_FFN_1.store_FFN_activation(y)

        x = self.up_proj(x)
        if self.calibration and self.FFN_act_score:
            self.prune_API_MLP_FFN_2.store_FFN_activation(x)

        x = x*y
        if self.calibration and self.FFN_act_score:
            self.prune_API_MLP_attention.store_FFN_activation(x)

        x = self.down_proj(x)
        if self.calibration and self.FFN_act_score:
            self.prune_API_MLP_FFN_3.store_FFN_activation(x)
        
        return x
    
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim,           bias=False)
        self.k_proj = nn.Linear(self.hidden_size,               self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size,               self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size,                         bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.static_kv_repeat = True

    def get_attn_hidden_size_score(self):

        hidden_size_score = 0

        if self.prune_hidden_size:
            hidden_size_score = self.prune_API_MHSA_o.get_FFN_layer_score() 

        return hidden_size_score


    def initialize_attn_pruning_functions(self):
        self.prune_API_MHSA_q    = PruneGPT_Attention(prune_dimension = self.num_heads)
        self.prune_API_MHSA_k    = PruneGPT_Attention(prune_dimension = self.num_heads)
        self.prune_API_MHSA_v    = PruneGPT_Attention(prune_dimension = self.num_heads)
        self.prune_API_MHSA_attn = PruneGPT_Attention(prune_dimension = self.num_heads)
        self.prune_API_MHSA_o    = PruneGPT_FFN_hidden_size(prune_dimension = self.num_heads)
        
    
    def save_attn_device(self):
        self.prune_API_MHSA_q.save_attn_device(device    = self.q_proj.weight.device)   
        self.prune_API_MHSA_k.save_attn_device(device    = self.k_proj.weight.device)   
        self.prune_API_MHSA_v.save_attn_device(device    = self.v_proj.weight.device)   
        self.prune_API_MHSA_attn.save_attn_device(device = self.q_proj.weight.device)
        self.prune_API_MHSA_o.save_FFN_device(device    = self.o_proj.weight.device)   
        

    def store_attn_weight_gradients(self):
        
        q_weight = self.q_proj.weight
        k_weight = self.repeat_kv_weight(self.k_proj.weight.clone().detach(), self.num_key_value_groups, self.num_key_value_heads, self.head_dim)
        v_weight = self.repeat_kv_weight(self.v_proj.weight.clone().detach(), self.num_key_value_groups, self.num_key_value_heads, self.head_dim)
        o_weight = self.o_proj.weight
        
        q_grad = self.q_proj.weight.grad
        k_grad = self.repeat_kv_weight(self.k_proj.weight.grad.clone().detach(), self.num_key_value_groups, self.num_key_value_heads,  self.head_dim)
        v_grad = self.repeat_kv_weight(self.v_proj.weight.grad.clone().detach(), self.num_key_value_groups, self.num_key_value_heads,  self.head_dim) 
        o_grad = self.o_proj.weight.grad


        self.prune_API_MHSA_q.store_attn_weight_gradients(q_weight, q_grad)
        self.prune_API_MHSA_k.store_attn_weight_gradients(k_weight, k_grad)
        self.prune_API_MHSA_v.store_attn_weight_gradients(v_weight, v_grad)
        self.prune_API_MHSA_o.store_FFN_weight_gradients(o_weight, o_grad)



    def store_attn_weights(self):

        self.prune_API_MHSA_q.store_attn_weights(self.q_proj.weight, self.num_heads, self.head_dim)
        
        self.prune_API_MHSA_k.store_attn_weights(self.repeat_kv_weight(self.k_proj.weight.clone().detach(), self.num_key_value_groups, self.num_key_value_heads, self.head_dim),
                                                  self.num_heads, self.head_dim)

        self.prune_API_MHSA_v.store_attn_weights(self.repeat_kv_weight(self.v_proj.weight.clone().detach(), self.num_key_value_groups, self.num_key_value_heads, self.head_dim), 
                                                  self.num_heads, self.head_dim)

        self.prune_API_MHSA_o.store_FFN_weights(self.o_proj.weight)


    def store_attn_gradients(self):
        
        k_proj_weight_grad = self.repeat_kv_weight(self.k_proj.weight.grad.clone().detach(), self.num_key_value_groups, self.num_key_value_heads,  self.head_dim)
        v_proj_weight_grad = self.repeat_kv_weight(self.v_proj.weight.grad.clone().detach(), self.num_key_value_groups, self.num_key_value_heads,  self.head_dim)

        self.prune_API_MHSA_q.store_attn_gradients(self.q_proj.weight.grad)
        self.prune_API_MHSA_k.store_attn_gradients(k_proj_weight_grad) 
        self.prune_API_MHSA_v.store_attn_gradients(v_proj_weight_grad)
        self.prune_API_MHSA_o.store_FFN_gradients(self.o_proj.weight.grad)


    def set_attn_calibration(self, calibration):
        self.calibration = calibration
        
    def set_attn_pruning_choices(self, args):
        self.prune_API_MHSA_q.set_attn_pruning_choices(args)
        self.prune_API_MHSA_k.set_attn_pruning_choices(args)
        self.prune_API_MHSA_v.set_attn_pruning_choices(args)
        self.prune_API_MHSA_attn.set_attn_pruning_choices(args)
        self.prune_API_MHSA_o.set_FFN_pruning_choices(args)
        
        self.prune_API_MHSA_o.FFN_act_score = False

        self.prune_Head        =  args.prune_Head
        self.prune_Heads_num   =  args.prune_Heads_num
        self.prune_hidden_size =  args.prune_hidden_size

        self.Attention_wt_score    =  args.Attention_wt_score
        self.Attention_act_score   =  args.Attention_act_score
        self.Attention_grad_score  =  args.Attention_grad_score

        self.wt_plus_grad_plus_act  = args.wt_plus_grad_plus_act
        self.wt_mult_grad_plus_act  = args.wt_mult_grad_plus_act
        
                
    def update_attn_activation_samples(self):
        self.prune_API_MHSA_q.update_attn_activation_samples()
        self.prune_API_MHSA_k.update_attn_activation_samples()
        self.prune_API_MHSA_v.update_attn_activation_samples()
        self.prune_API_MHSA_attn.update_attn_activation_samples()
        self.prune_API_MHSA_o.update_FFN_activation_samples()
        

    def repeat_kv_dynamic(self, hidden_states, n_reps, num_key_value_heads, head_dim):

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states.squeeze(0).transpose(0, 1)
        reshaped_weight = hidden_states.view(slen, num_key_value_heads, head_dim)

        repeated_tensors = []

        for i in range(num_key_value_heads):
            repeated_tensors.append(reshaped_weight[:, i, :].repeat(1, n_reps[i]))

        repeated_tensors = torch.cat(repeated_tensors, dim=1)

        repeated_tensors = repeated_tensors.view(slen, sum(n_reps), head_dim)
        repeated_tensors = repeated_tensors.transpose(0, 1).unsqueeze(0)
        
        return repeated_tensors


    def repeat_kv_weight(self, kv_weight, n_rep, num_key_value_heads, head_dim):

        kv_weight = torch.transpose(kv_weight, 0, 1)
        input_neurons, output_neurons = kv_weight.shape
        kv_weight = kv_weight.view(input_neurons, num_key_value_heads, head_dim).transpose(0, 1).contiguous()
        
        if n_rep == 1:
            assert output_neurons == num_key_value_heads*head_dim
            kv_weight = torch.transpose(kv_weight, 0, 1)
            return kv_weight
        
        else:
            kv_weight = kv_weight[:, None, :, :].expand(num_key_value_heads, n_rep, input_neurons, head_dim)
            kv_weight = kv_weight.reshape(num_key_value_heads * n_rep, input_neurons, head_dim)
            kv_weight = torch.transpose(kv_weight, 0, 1)
            kv_weight = kv_weight.reshape(input_neurons, num_key_value_heads*n_rep*head_dim)
            kv_weight = torch.transpose(kv_weight, 0, 1)
            return kv_weight


    def get_kv_mask(self, query_prune_mask, num_groups):

        kv_prune_mask_list = []
        for i in range(0, len(query_prune_mask), num_groups):
            chunk = query_prune_mask[i:i+num_groups]
            kv_prune_mask_list.append(chunk.any())

        kv_mask = torch.tensor(kv_prune_mask_list, dtype=torch.bool)
        return kv_mask



    def dynamic_expansion(self, prune_mask_tensor, num_groups):
        reshaped_tensor = prune_mask_tensor.view(-1, num_groups)
        count_tensor = reshaped_tensor.sum(dim=1)
        non_zero_mask = count_tensor != 0
        result = count_tensor[non_zero_mask]
        return result

        
    def prune_attn_fc_layer(self, global_hidden_size_prune_mask):
        
        if self.prune_hidden_size:
            
            if self.layer_idx != 0:
                self.q_proj = prune_input_linear_layer(self.q_proj, global_hidden_size_prune_mask)
                self.k_proj = prune_input_linear_layer(self.k_proj, global_hidden_size_prune_mask)
                self.v_proj = prune_input_linear_layer(self.v_proj, global_hidden_size_prune_mask)
            
            self.o_proj = prune_output_linear_layer(self.o_proj, global_hidden_size_prune_mask)
            
            self.hidden_size = sum(global_hidden_size_prune_mask)



        if self.prune_Head:
            
            dict_args = {"num_heads": self.num_heads, 
                         "head_dim":  self.head_dim}
            
            q_score_head = self.prune_API_MHSA_q.get_attn_layer_score(**dict_args) 
            k_score_head = self.prune_API_MHSA_k.get_attn_layer_score(**dict_args) 
            v_score_head = self.prune_API_MHSA_v.get_attn_layer_score(**dict_args)
            
            attn_head_score = self.prune_API_MHSA_attn.get_head_act_score(**dict_args)
            
            qkv_avg = get_qkv_avg(q_score_head, k_score_head, v_score_head)

            assert attn_head_score.shape == qkv_avg.shape 

            qkv_score_head =  attn_head_score + qkv_avg

            prune_mask_q_head   = get_prune_mask_from_score_scalar(qkv_score_head, self.prune_Heads_num)
            prune_mask_q   = get_linear_prune_mask(prune_mask_q_head, self.head_dim)

            if self.num_key_value_groups == 1:
                prune_mask_kv = prune_mask_q
                self.static_kv_repeat = True

            else:
                prune_mask_kv_head  = self.get_kv_mask(prune_mask_q_head, self.num_key_value_groups)
                prune_mask_kv       = get_linear_prune_mask(prune_mask_kv_head, self.head_dim)
                self.n_reps         = self.dynamic_expansion(prune_mask_q_head, self.num_key_value_groups)
                self.static_kv_repeat = False


            self.q_proj    = prune_output_linear_layer(self.q_proj,  prune_mask_q)
            self.k_proj    = prune_output_linear_layer(self.k_proj, prune_mask_kv)
            self.v_proj    = prune_output_linear_layer(self.v_proj, prune_mask_kv)
            self.o_proj     = prune_input_linear_layer(self.o_proj,  prune_mask_q)
            
            self.num_key_value_heads = sum(prune_mask_kv_head)
            self.num_heads           = sum(prune_mask_q_head) 

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads

        if self.static_kv_repeat:
            key_states   = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        else:
            key_states   = self.repeat_kv_dynamic(key_states, self.n_reps, self.num_key_value_heads, self.head_dim)
            value_states = self.repeat_kv_dynamic(value_states, self.n_reps, self.num_key_value_heads, self.head_dim)


        if self.calibration and self.Attention_act_score:
            self.prune_API_MHSA_q.store_attn_activation(query_states)
            self.prune_API_MHSA_k.store_attn_activation(key_states)
            self.prune_API_MHSA_v.store_attn_activation(value_states)


        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if self.calibration:
            self.prune_API_MHSA_attn.store_attn_activation(attn_output)


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = attn_output.reshape(bsz, q_len, int(self.num_heads*self.head_dim))

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # if self.calibration and self.Attention_act_score:
        #     self.prune_API_MHSA_o.update_activation(attn_output)

        return attn_output, attn_weights, past_key_value



class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()

        self.hidden_size              = config.hidden_size
        self.layer_idx                = layer_idx
        
        self.self_attn                = MistralAttention(config, layer_idx)
        self.mlp                      = MistralMLP(config)

        self.input_layernorm          = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def initialize_pruning_functions(self):

        self.prune_API_MHSA_attn = PruneGPT_FFN_hidden_size(prune_dimension = None)
        self.prune_API_MLP_attn  = PruneGPT_FFN_hidden_size(prune_dimension = None)
        
        self.self_attn.initialize_attn_pruning_functions()
        self.mlp.initialize_FFN_pruning_functions()
    
    def store_weight_gradients(self):
        self.self_attn.store_attn_weight_gradients()
        self.mlp.store_FFN_weight_gradients()

    def store_gradients(self):
        self.self_attn.store_attn_gradients()
        self.mlp.store_FFN_gradients()


    def store_weights(self):
        self.self_attn.store_attn_weights()
        self.mlp.store_FFN_weights()



    def get_hidden_size_score(self):
        hidden_size_score = 0

        if self.prune_hidden_size:
            hidden_size_score += self.self_attn.get_attn_hidden_size_score() 
            hidden_size_score += self.mlp.get_FFN_hidden_size_score() 
        
            if self.Attention_act_score:
                hidden_size_score += self.prune_API_MHSA_attn.get_FFN_layer_score()

            if self.FFN_act_score:
                hidden_size_score += self.prune_API_MLP_attn.get_FFN_layer_score()

        return hidden_size_score


    def prune_fc_layer(self, global_hidden_size_prune_mask):
        self.global_hidden_size_prune_mask = global_hidden_size_prune_mask
        if self.prune_hidden_size and self.layer_idx != 0:
            self.input_layernorm.prune_weight(global_hidden_size_prune_mask)
        
        self.self_attn.prune_attn_fc_layer(global_hidden_size_prune_mask)
        self.mlp.prune_FFN_fc_layer(global_hidden_size_prune_mask)
        
        if self.prune_hidden_size:
            self.post_attention_layernorm.prune_weight(global_hidden_size_prune_mask)
    
    def update_activation_samples(self):
        self.self_attn.update_attn_activation_samples()
        self.mlp.update_FFN_activation_samples()
    
    def save_device(self):

        assert self.self_attn.q_proj.weight.device == self.mlp.gate_proj.weight.device
        assert self.self_attn.q_proj.weight.device == self.input_layernorm.weight.device
        assert self.self_attn.q_proj.weight.device == self.post_attention_layernorm.weight.device

        self.prune_API_MHSA_attn.save_FFN_device(self.self_attn.q_proj.weight.device)
        self.prune_API_MLP_attn.save_FFN_device(self.self_attn.q_proj.weight.device)

        self.self_attn.save_attn_device()
        self.mlp.save_FFN_device()
        
    def set_calibration(self, calibration):
        self.self_attn.set_attn_calibration(calibration)
        self.mlp.set_FFN_calibration(calibration)
        self.calibration = calibration
        
    def set_pruning_choices(self, args):
        self.prune_hidden_size = args.prune_hidden_size
        
        self.Attention_act_score = args.Attention_act_score
        self.FFN_act_score = args.FFN_act_score
        
        self.self_attn.set_attn_pruning_choices(args)
        self.mlp.set_FFN_pruning_choices(args)

        self.prune_API_MHSA_attn.set_FFN_pruning_choices(args)
        self.prune_API_MLP_attn.set_FFN_pruning_choices(args)

        self.prune_API_MHSA_attn.FFN_wt_score = False
        self.prune_API_MHSA_attn.FFN_grad_score = False

        self.prune_API_MLP_attn.FFN_wt_score = False
        self.prune_API_MLP_attn.FFN_grad_score = False

        self.prune_API_MHSA_attn.wt_mult_grad_plus_act = False
        self.prune_API_MHSA_attn.wt_plus_grad_plus_act = False

        self.prune_API_MLP_attn.wt_mult_grad_plus_act = False
        self.prune_API_MLP_attn.wt_plus_grad_plus_act = False

        
    def prune_activations(self, act, idx):
        if not self.calibration:
            if self.prune_hidden_size and idx == 0:
                self.global_hidden_size_prune_mask = self.global_hidden_size_prune_mask.to(device = act.device)
                act = act[:, :, self.global_hidden_size_prune_mask]
        return act


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # hidden_states = residual + hidden_states
        hidden_states =  self.prune_activations(residual, self.layer_idx) + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.calibration:
            self.prune_API_MHSA_attn.store_FFN_activation(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.calibration:
            self.prune_API_MLP_attn.store_FFN_activation(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class MistralPreTrainedModel(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



class MistralModel(MistralPreTrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        self.layers = nn.ModuleList([MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        
        self._attn_implementation = config._attn_implementation

        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        
        self.post_init()


    def get_hidden_size_score(self):

        hidden_size_score = 0

        if self.prune_hidden_size:
            for layer in self.layers:
                hidden_size_score += layer.get_hidden_size_score() 

        return hidden_size_score


    def initialize_pruning_functions(self):
        for layer in self.layers:
            layer.initialize_pruning_functions()
    
    def store_gradients(self):
        for layer in self.layers:
            layer.store_gradients()

    
    def store_weights(self):
        for layer in self.layers:
            layer.store_weights()
    

    def store_weight_gradients(self):
        for layer in self.layers:
            layer.store_weight_gradients()

    def update_activation_samples(self):
        for layer in self.layers:
            layer.update_activation_samples()
    
    def prune_fc_layer(self, global_hidden_size_prune_mask):
        
        for layer in self.layers:
            layer.prune_fc_layer(global_hidden_size_prune_mask)
            
        if self.prune_hidden_size:
            self.norm.prune_weight(global_hidden_size_prune_mask)
            
            
    def save_device(self):
        for layer in self.layers:
            layer.save_device()
    
    def set_calibration(self, calibration):
        for layer in self.layers:
            layer.set_calibration(calibration)
    
    def set_pruning_choices(self, args):
        for layer in self.layers:
            layer.set_pruning_choices(args)
        self.prune_hidden_size = args.prune_hidden_size


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def initialize_pruning_functions(self):
        self.model.initialize_pruning_functions()

    def update_activation_samples(self):
        self.model.update_activation_samples()

    def get_hidden_size_score(self):

        hidden_size_score = 0

        if self.prune_hidden_size:
            hidden_size_score = self.model.get_hidden_size_score() 

        return hidden_size_score


    def get_global_hidden_size_prune_mask(self):

        if self.prune_hidden_size:
            hidden_size_score = self.get_hidden_size_score()

            # prune_nodes = int(np.round((len(hidden_size_score)*self.prune_hidden_size_sparsity)/(16*100))*16)

            global_hidden_size_prune_mask = get_prune_mask_from_score(len(hidden_size_score), 
                                                                  hidden_size_score, 
                                                                  self.prune_hidden_size_sparsity, 
                                                                  16)
        else:
            global_hidden_size_prune_mask = None
        
        return global_hidden_size_prune_mask


    def prune_fc_layer(self):

        global_hidden_size_prune_mask = self.get_global_hidden_size_prune_mask()
        self.model.prune_fc_layer(global_hidden_size_prune_mask)
        
        if self.prune_hidden_size:
            self.lm_head = prune_input_linear_layer(self.lm_head, global_hidden_size_prune_mask)
        
    def save_device(self):
        self.model.save_device()
        
    def set_calibration(self, calibration):
        self.model.set_calibration(calibration)
        
    def set_pruning_choices(self, args):
        self.model.set_pruning_choices(args)
        self.prune_hidden_size = args.prune_hidden_size
        self.prune_hidden_size_sparsity = args.prune_hidden_size_sparsity

    def store_weight_grad(self):
        self.model.store_weight_gradients()

    def store_grads(self):
        self.model.store_gradients()

    def store_weights(self):
        self.model.store_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past



if __name__ == '__main__':
    
    mistral_model = "mistralai/Mistral-7B-v0.1"
    
    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

    Mistral_model = MistralForCausalLM.from_pretrained(mistral_model, cache_dir = cache_dir, device_map = "auto")
    
    print("model loaded")
    
    input_tensor  = torch.rand((1, 512)).to(dtype=torch.int8).long()
    output_tensor = Mistral_model(input_tensor)

    print("output_tensor.shape", output_tensor[0].shape)
