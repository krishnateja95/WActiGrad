import sys, os
import json
import torch
import numpy as np

def print_requires_grad_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

                
def get_LLM(model_name):
    if "llama" in model_name:
        from modeling_sparse_llama import LlamaForCausalLM
        from transformers import LlamaTokenizer

        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, device_map  = "auto")
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        

    elif "Mistral" in model_name:
        from modeling_sparse_mistral import MistralForCausalLM
        from transformers import AutoTokenizer

        model = MistralForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, device_map  = "auto")
        tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side='left'
    
    model.seqlen = 2048
    return model, tokenizer 
    
    
    
