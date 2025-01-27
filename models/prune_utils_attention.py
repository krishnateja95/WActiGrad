import time
import copy
import torch
import argparse
import numpy as np
import transformers
import torch.nn as nn
from torch.nn import functional as F

from prune_utils_general import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

def get_qkv_avg(q, k, v):
    return torch.stack([q, k, v]).mean(dim = 0)


class PruneGPT_Attention():
    def __init__(self, prune_dimension):
        super().__init__()
         
        self.prune_dimension = prune_dimension
        
        self.activation_samples =  0
        
        self.attn_weight_updated           =  False
        self.attn_activations_updated      =  False
        self.attn_gradient_updated         =  False
        self.attn_weight_gradient_updated  =  False
        
    def update_attn_activation_samples(self):    
        self.activation_samples += 1
        
    def get_attn_activation_samples(self):    
        return self.activation_samples
    
    def set_attn_pruning_choices(self, args):
        
        self.l1_norm     =  args.l1_norm
        self.l2_norm     =  args.l2_norm
        
        self.Attention_wt_score    =  args.Attention_wt_score
        self.Attention_act_score   =  args.Attention_act_score
        self.Attention_grad_score  =  args.Attention_grad_score
        
        self.model_name      =  args.model
        self.normalizer      =  args.normalize

        self.wt_plus_grad_plus_act  =  args.wt_plus_grad_plus_act
        self.wt_mult_grad_plus_act  =  args.wt_mult_grad_plus_act
        
    def save_attn_device(self, device):
        self.device    =  device

    
    def get_head_score(self, tensor, num_heads, head_dim):
        
        tensor = tensor.clone().detach().to(self.device)
        
        assert len(tensor.shape) == 2
        
        input_neurons, output_neurons = tensor.shape
        tensor = tensor.view(input_neurons, num_heads, head_dim).transpose(0, 1).contiguous()
        
        head_score = torch.zeros((num_heads)).to(device = self.device)

        for idx, head in enumerate(tensor):
            head_score[idx] = torch.norm(head, p = 2)

        head_score =  normalize_scores(head_score, self.normalizer)
            
        return head_score
    


    def get_head_score_activation(self, tensor, num_heads, head_dim):
        
        tensor = tensor.clone().detach().to(self.device)
        assert len(tensor.shape) == 3
        assert list(tensor.shape) == [num_heads, 2048, head_dim]

        head_score = torch.zeros((num_heads)).to(device = self.device)

        for idx, head in enumerate(tensor):
            head_score[idx] = torch.norm(head, p = 2)

        head_score = normalize_scores(head_score, self.normalizer)
            
        return head_score
    
    
    def store_attn_weights(self, weight, num_heads, head_dim):
        self.attn_weight_updated = True
        self.weight_tensor = weight.clone().detach().to(self.device)
        self.weight_tensor = torch.transpose(self.weight_tensor, 0, 1)
        self.weight_score_head = self.get_head_score(self.weight_tensor.clone().detach(), num_heads, head_dim)
       

    def store_attn_weight_gradients(self, attn_weight, attn_grad):
        
        attn_grad   = attn_grad.clone().detach().t().to(device = self.device)
        attn_weight = attn_weight.clone().detach().t().to(device = self.device)
        
        if self.wt_mult_grad_plus_act:
            attn_weight_grad = torch.abs(attn_weight*attn_grad)

        if self.attn_weight_gradient_updated:
            self.attn_weight_grad_avg = (self.attn_weight_grad_avg * self.activation_samples + attn_weight_grad) / (self.activation_samples + 1)
        else:
            self.attn_weight_grad_avg = attn_weight_grad
            
        self.attn_weight_gradient_updated = True



    def get_weight_grad_attn_score(self, num_heads, head_dim):
        if not self.attn_weight_gradient_updated:
            raise ValueError("Weights and Gradients not updated. Exiting")
            
        weight_grad_score_head = self.get_head_score(self.attn_weight_grad_avg, num_heads, head_dim)
        return weight_grad_score_head 

    def store_attn_gradients(self, grad):
        grad = grad.clone().detach().t().to(device = self.device)
        if self.attn_gradient_updated:
            self.grad_avg = (self.grad_avg * self.activation_samples + grad) / (self.activation_samples + 1)
        else:
            self.grad_avg = grad
        self.attn_gradient_updated = True
        


    def get_attn_gradients_score(self, num_heads, head_dim):
        if not self.attn_gradient_updated:
            if not self.attn_weight_updated:
                raise ValueError("Gradients not updated. Exiting")
            
        grad_score_head = self.get_head_score(self.grad_avg, num_heads, head_dim)
        return grad_score_head 

        
    def store_attn_activation(self, act):
        act = act.clone().to(device = self.device)
        act = act.clone().detach().squeeze(0)
        if self.attn_activations_updated:
            self.act_avg = (self.act_avg * self.activation_samples + act) / (self.activation_samples + 1)
        else:
            self.act_avg = act
        self.attn_activations_updated = True

         

    def get_attn_activation_score(self, num_heads, head_dim):
        if not self.attn_activations_updated:
            raise ValueError("Activations not updated. Exiting")
        activation_score_head = self.get_head_score_activation(self.act_avg, num_heads, head_dim)
        return activation_score_head
    
    
    def get_head_act_score(self, num_heads, head_dim):
        activation_score_head = self.get_attn_activation_score(num_heads, head_dim)
        return activation_score_head 
    
    
    def get_attn_layer_score(self, num_heads, head_dim):
        score_head = 0
        if self.Attention_act_score:
            assert self.attn_activations_updated == True
            activation_score_head = self.get_attn_activation_score(num_heads, head_dim)

            if self.Attention_wt_score:
                if self.wt_plus_grad_plus_act:
                    assert self.weight_score_head.shape == activation_score_head.shape
                elif self.wt_mult_grad_plus_act:
                    assert self.get_weight_grad_attn_score(num_heads, head_dim).shape == activation_score_head.shape
            score_head += activation_score_head


        if self.wt_mult_grad_plus_act:
            assert self.attn_weight_gradient_updated == True
            score_head += self.get_weight_grad_attn_score(num_heads, head_dim)


        elif self.wt_plus_grad_plus_act:
            if self.Attention_wt_score:
                assert self.attn_weight_updated == True
                score_head += self.weight_score_head 
                
            if self.Attention_grad_score:
                assert self.attn_gradient_updated == True
                grad_score_head = self.get_attn_gradients_score(num_heads, head_dim)

                if self.Attention_wt_score:
                    assert self.weight_score_head.shape == grad_score_head.shape
                score_head += grad_score_head
                
        return score_head
    
    