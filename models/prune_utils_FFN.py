
import time
import copy
import torch
import argparse
import numpy as np
import transformers
import torch.nn as nn

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../../../')

from models.prune_utils_general import *


from eval.data import get_loaders
from torch.nn import functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False


class PruneGPT_FFN_hidden_size():
    def __init__(self, prune_dimension):
        super().__init__()
         
        self.prune_dimension = prune_dimension
        
        self.activation_samples =  0
        
        self.FFN_weight_updated          =  False
        self.FFN_activations_updated     =  False
        self.FFN_gradient_updated        =  False
        self.FFN_weight_gradient_updated =  False
        
    def update_FFN_activation_samples(self):    
        self.activation_samples += 1
        
    def get_FFN_activation_samples(self):    
        return self.activation_samples
    
    def set_FFN_pruning_choices(self, args):
        
        self.l1_norm     =  args.l1_norm
        self.l2_norm     =  args.l2_norm
        
        self.FFN_wt_score    =  args.FFN_wt_score
        self.FFN_act_score   =  args.FFN_act_score
        self.FFN_grad_score  =  args.FFN_grad_score
        
        self.model_name      =  args.model
        self.normalizer      =  args.normalize

        self.wt_plus_grad_plus_act = args.wt_plus_grad_plus_act
        self.wt_mult_grad_plus_act = args.wt_mult_grad_plus_act
        
    def save_FFN_device(self, device):
        self.device    =  "cpu"
        # self.device    =  device
        


    def get_neuron_score(self, tensor):
        
        tensor = tensor.clone().detach().to(self.device)
        
        output_neurons = tensor.shape[1]
        neuron_score = torch.zeros((output_neurons)).to(device = self.device)
        
        if self.l1_norm:
            # neuron_score += self.normalize_scores(self.l1_norm_score(tensor, dim = 0))
            neuron_score =  normalize_scores(l1_norm_score(tensor, dim = 0), self.normalizer)

        if self.l2_norm: 
            neuron_score =  normalize_scores(l2_norm_score(tensor, dim = 0), self.normalizer)
                
        return neuron_score


    
    
    def store_FFN_weights(self, weight):
        
        self.FFN_weight_updated = True
        
        self.weight_tensor = weight.clone().detach().to(self.device)
        self.weight_tensor = torch.transpose(self.weight_tensor, 0, 1)
        
        self.weight_score_neuron = self.get_neuron_score(self.weight_tensor.clone().detach())


    def store_FFN_gradients(self, grad):
        
        grad = grad.clone().detach().t().to(device = self.device)
        
        if self.FFN_gradient_updated:
            self.grad_avg = (self.grad_avg * self.activation_samples + grad) / (self.activation_samples + 1)
        else:
            self.grad_avg = grad
        self.FFN_gradient_updated = True
    

    def store_FFN_weight_gradients(self, FFN_weight, FFN_grad):
        
        FFN_grad   = FFN_grad.clone().detach().t().to(device = self.device)
        FFN_weight = FFN_weight.clone().detach().t().to(device = self.device)
        
        if self.wt_mult_grad_plus_act:
            FFN_weight_grad = torch.abs(FFN_weight*FFN_grad)

        if self.FFN_weight_gradient_updated:
            self.weight_grad_avg = (self.weight_grad_avg * self.activation_samples + FFN_weight_grad) / (self.activation_samples + 1)
        else:
            self.weight_grad_avg = FFN_weight_grad
            
        self.FFN_weight_gradient_updated = True


    def store_FFN_activation(self, act):
        
        act = act.clone().to(device = self.device)
        act = act.clone().detach().squeeze(0)
        
        if self.FFN_activations_updated:
            self.act_avg = (self.act_avg * self.activation_samples + act) / (self.activation_samples + 1)
            
        else:
            self.act_avg = act
        
        self.FFN_activations_updated = True



    def get_weight_grad_FFN_score(self):
        
        if not self.FFN_weight_gradient_updated:
            print("Weight Gradients are not updated. Exiting")
            exit()
        
        grad_score_neuron = self.get_neuron_score(self.weight_grad_avg)
        return grad_score_neuron

    
    def get_FFN_gradients_score(self):
        
        if not self.FFN_gradient_updated:
            if not self.FFN_weight_updated:
                print("Gradients are not updated. Exiting")
                exit()
            
        weight_grad_score_neuron = self.get_neuron_score(self.grad_avg)
        return weight_grad_score_neuron 

    
    def get_FFN_activation_score(self):
        
        if not self.FFN_activations_updated:
            print("Activations not updated. Exiting")
            exit()
            
        activation_score_neuron = self.get_neuron_score(self.act_avg)
        
        return activation_score_neuron




    def get_FFN_layer_score(self):
        
        score_neuron = 0
        
        if self.FFN_act_score:
            assert self.FFN_activations_updated == True
            activation_score_neuron = self.get_FFN_activation_score()

            if self.FFN_wt_score:
                if self.wt_plus_grad_plus_act:
                    assert self.weight_score_neuron.shape == activation_score_neuron.shape
                elif self.wt_mult_grad_plus_act:
                    assert self.get_weight_grad_FFN_score().shape == activation_score_neuron.shape
            
            score_neuron += activation_score_neuron


        if self.wt_mult_grad_plus_act:
            assert self.FFN_weight_gradient_updated == True
            score_neuron += self.get_weight_grad_FFN_score()


        elif self.wt_plus_grad_plus_act:
            if self.FFN_wt_score:
                assert self.FFN_weight_updated == True
                score_neuron += self.weight_score_neuron

            if self.FFN_grad_score:
                assert self.FFN_gradient_updated == True
                grad_score_neuron = self.get_FFN_gradients_score()

                if self.FFN_wt_score:
                    assert self.weight_score_neuron.shape == grad_score_neuron.shape
                score_neuron += grad_score_neuron
        
        return score_neuron
    
    
    
    
    def get_FFN_layer_prune_mask(self, sparsity, prune_dim_nearest, attention, num_heads, head_dim):
        layer_score   =  self.get_layer_score(attention, num_heads, head_dim)
        prune_nodes   =  np.round((self.prune_dimension*sparsity)/(prune_dim_nearest*100))*prune_dim_nearest 
        thresh        =  torch.sort(layer_score.flatten())[0][int(prune_nodes)]
        mask          =  layer_score >= thresh

        # print("score", score)
        # print("thresh", thresh)

        print("prune_dimension", self.prune_dimension)
        print("sparsity", sparsity)
        print("prune_dim_nearest", prune_dim_nearest)
        print("prune_nodes", prune_nodes)
        print("bool mask", sum(mask))
        print("True Nodes", self.prune_dimension-prune_nodes)

        return mask
        
    
    