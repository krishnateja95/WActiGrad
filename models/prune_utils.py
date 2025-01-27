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

from eval.data import get_loaders
from torch.nn import functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

def prune_moe(args, model, tokenizer, device, nsamples_calibration, bs, calibration_dataset):
    
    trainloader, testloader = get_loaders(calibration_dataset,
                                          seed = 0,
                                          seqlen = model.seqlen,
                                          tokenizer = tokenizer)

    print(f"Calibration using {calibration_dataset}")
    print(f"Number of Training Samples =  {len(trainloader)}")
    print(f"Number of Calibration Samples =  {nsamples_calibration}")
    
    model.set_calibration(calibration = True)
    model.initialize_pruning_functions()
    model.set_pruning_choices(args)
    model.save_device()
    
    
    for i in range(0, nsamples_calibration, bs):
        print("Sample ", i)
        j = min(i+bs, len(trainloader))

        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss     = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        
        loss.backward()
        
        if model.wt_plus_grad_plus_act:
            model.store_grads()
        else:
            model.store_weight_grad()


        model.update_activation_samples()

    if model.wt_plus_grad_plus_act:
        model.store_weights()

    torch.cuda.empty_cache()
    
    model.prune_fc_layer()
    model.set_calibration(calibration = False)    
    
    _ = model(inputs).logits

    print("End of Calibration")

    return model 

def get_qkv_avg(q, k, v):
    return torch.stack([q, k, v]).mean(dim = 0)

def get_prune_mask_from_score_scalar(score, prune_nodes):
    thresh        =  torch.sort(score.flatten())[0][int(prune_nodes)]
    mask          =  score >= thresh
    return mask

def get_prune_mask_from_score(prune_dimension, score, sparsity, prune_dim_nearest):
    prune_nodes   =  int(np.round((prune_dimension*sparsity)/(prune_dim_nearest*100))*prune_dim_nearest)
    thresh        =  torch.sort(score.flatten())[0][int(prune_nodes)]
    mask          =  score >= thresh
    
    # print("score", score)
    # print("thresh", thresh)

    # print("prune_dimension", prune_dimension)
    # print("sparsity", sparsity)
    # print("prune_dim_nearest", prune_dim_nearest)
    # print("prune_nodes", prune_nodes)
    # print("bool mask", sum(mask))
    # print("True Nodes", prune_dimension-prune_nodes)

    return mask


def get_linear_prune_mask(prune_mask_qkv_head, head_dim):
    prune_mask_qkv_head = prune_mask_qkv_head.unsqueeze(1).repeat(1, head_dim).view(-1)
    return prune_mask_qkv_head


def get_overall_prune_mask(prune_mask_qkv_head, prune_mask_qkv_head_dim, prune_Head, prune_Head_Dim, num_heads, head_dim):
    
    pruned_heads    = sum(prune_mask_qkv_head)
    pruned_head_dim = sum(prune_mask_qkv_head_dim)
    
    prune_mask_qkv_head = prune_mask_qkv_head.unsqueeze(1).repeat(1, head_dim).view(-1)
    
    repeated_num_heads      = prune_mask_qkv_head_dim.repeat(num_heads)
    prune_mask_qkv_head_dim = torch.cat((prune_mask_qkv_head_dim, repeated_num_heads), dim=0)
    
    if prune_Head and prune_Head_Dim:
        prune_mask_qkv = torch.tensor([True if m1*m2==1 else False for m1, m2 in zip(prune_mask_qkv_head, prune_mask_qkv_head_dim)],
                                      dtype=torch.bool)
        return prune_mask_qkv, pruned_heads, pruned_head_dim
    
    elif prune_Head and not prune_Head_Dim:
        return prune_mask_qkv_head, pruned_heads, head_dim 

    elif not prune_Head and prune_Head_Dim:
        return prune_mask_qkv_head_dim, num_heads, pruned_head_dim
    

    
def prune_output_linear_layer(linear_layer, prune_mask):
    
    assert linear_layer.weight.shape[0] == prune_mask.shape[0]
    
    device = linear_layer.weight.device
    
    prune_mask = prune_mask.to(device)
    
    linear_layer.weight.data = linear_layer.weight[prune_mask,:].to(device)
    
    if linear_layer.bias is not None:
        linear_layer.bias.data = (linear_layer.bias*prune_mask).to(device) 
    
    linear_layer.out_features = linear_layer.weight.shape[0]
    
    return linear_layer.to(device)



def prune_input_linear_layer(linear_layer, prune_mask):
    
    assert linear_layer.weight.shape[1] == prune_mask.shape[0]
           
    device = linear_layer.weight.device
    prune_mask = prune_mask.to(device)
     
    linear_layer.weight.data = linear_layer.weight[:,prune_mask].to(device)
    linear_layer.in_features = linear_layer.weight.shape[-1]
    
    return linear_layer.to(device)
    
            

def prune_norm_layer(norm_layer, prune_mask):
    
    assert norm_layer.weight.shape[0] == prune_mask.shape[0]
    
    device         = norm_layer.weight.device
    prune_mask     = prune_mask.to(device)
    
    norm_layer.weight.data = (norm_layer.weight * prune_mask[None,:]).squeeze(0).to(device)
    norm_layer.bias.data = (norm_layer.bias * prune_mask[None,:]).squeeze(0).to(device)

    return norm_layer.to(device)
    




# class PruneGPT_Attention():
#     def __init__(self, num_heads):
#         super().__init__()
         
#         self.num_heads = num_heads
        
#         self.activation_samples =  0
        
#         self.wts_updated  =  False
#         self.acts_updated =  False
#         self.grad_updated =  False


#     def update_activation_samples(self):    
#         self.activation_samples += 1
        
#     def get_activation_samples(self):    
#         return self.activation_samples
    


#     def store_attn_weight_gradients(self, attn_weight, attn_grad):
        
#         attn_grad   = attn_grad.clone().detach().t().to(device = self.device)
#         attn_weight = attn_weight.clone().detach().t().to(device = self.device)
        
#         attn_weight_grad = torch.abs(attn_weight)*torch.abs(attn_grad)
        
#         if self.attn_weight_grad_updated:
#             self.attn_weight_grad_avg = (self.attn_weight_grad_avg * self.activation_samples + attn_weight_grad) / (self.activation_samples + 1)
#         else:
#             self.attn_weight_grad_avg = attn_weight_grad
            
#         self.attn_weight_grad_updated = True









# class PruneGPT():
#     def __init__(self, prune_dimension):
#         super().__init__()
         
#         self.prune_dimension = prune_dimension
        
#         self.activation_samples =  0
        
#         self.wts_updated  =  False
#         self.acts_updated =  False
#         self.grad_updated =  False
        
#     def update_activation_samples(self):    
#         self.activation_samples += 1
        
#     def get_activation_samples(self):    
#         return self.activation_samples
    
#     def set_pruning_choices(self, args):
        
#         self.l1_norm     =  args.l1_norm
#         self.l2_norm     =  args.l2_norm
        
#         self.Attention_wt_score    =  args.Attention_wt_score
#         self.Attention_act_score   =  args.Attention_act_score
#         self.Attention_grad_score  =  args.Attention_grad_score
        
#         self.FFN_wt_score    =  args.FFN_wt_score
#         self.FFN_act_score   =  args.FFN_act_score
#         self.FFN_grad_score  =  args.FFN_grad_score
        
#         self.model_name      = args.model
#         self.normalizer      =  args.normalize

#         self.wt_plus_grad_plus_act = args.wt_plus_grad_plus_act
#         self.wt_mult_grad_plus_act = args.wt_mult_grad_plus_act
        
#     def save_device(self, device):
#         self.device    =  "cpu"
#         # self.device    =  device
        
    
#     def compute_importance(self, *score_args, prune_dimension, combine_type = "prod"):
#         if combine_type == "sum":
#             combined_score = torch.zeros((self.prune_dimension))
#         elif combine_type == "sum":
#             combined_score = torch.ones((self.prune_dimension))
#         else:
#             raise NotImplementedError
        
#         for score in score_args:
#             if combine_type == "sum":
#                 combined_score += score
#             elif combine_type == "sum":
#                 combined_score *= score
#             else:
#                 raise NotImplementedError

#         return combined_score.to(device = self.device)
    
#     def normalize_scores(self, score):
#         if self.normalizer == "softmax":
#             return F.softmax(score, dim = -1)
#         elif self.normalizer == "sum":
#             return score / score.sum()
#         elif self.normalizer == "standarization":
#             return (score - score.min()) / (score.max() - score.min()+1e-8)
#         elif self.normalizer == "mean":
#             return score / score.mean()
#         elif self.normalizer == "max":
#             return score / score.max()
#         elif self.normalizer == 'gaussian':
#             return (score - score.mean()) / (score.std()+1e-8)
#         else:
#             raise NotImplementedError
    

#     def get_head_dim_score(self, tensor, num_heads, head_dim):

#         assert len(tensor.shape) == 2
#         tensor = tensor.clone().detach().to(self.device)
#         input_neurons, output_neurons = tensor.shape
        
#         neuron_score = self.l2_norm_score(tensor, dim = 0)
        
#         for i in range(0, output_neurons, head_dim):
#             neuron_score[i:i+head_dim] = self.normalize_scores(neuron_score[i:i+head_dim]) 
        
# #         head_dim_score = neuron_score.view(num_heads, head_dim)
#         head_dim_score = head_dim_score.sum(0)
#         return head_dim_score



#     def store_attn_weight_gradients(self, attn_weight, attn_grad):
        
#         attn_grad   = attn_grad.clone().detach().t().to(device = self.device)
#         attn_weight = attn_weight.clone().detach().t().to(device = self.device)
        
#         attn_weight_grad = torch.abs(attn_weight)*torch.abs(attn_grad)
        
#         if self.attn_weight_grad_updated:
#             self.attn_weight_grad_avg = (self.attn_weight_grad_avg * self.activation_samples + attn_weight_grad) / (self.activation_samples + 1)
#         else:
#             self.attn_weight_grad_avg = attn_weight_grad
            
#         self.attn_weight_grad_updated = True



#     def store_FFN_weight_gradients(self, FFN_weight, FFN_grad):
        
#         FFN_grad   = FFN_grad.clone().detach().t().to(device = self.device)
#         FFN_weight = FFN_weight.clone().detach().t().to(device = self.device)
        
#         weight_grad = torch.abs(FFN_weight)*torch.abs(FFN_grad)
        
#         if self.weight_grad_updated:
#             self.weight_grad_avg = (self.weight_grad_avg * self.activation_samples + weight_grad) / (self.activation_samples + 1)
#         else:
#             self.weight_grad_avg = weight_grad
            
#         self.weight_grad_updated = True







#     def get_neuron_score(self, tensor):
        
#         tensor = tensor.clone().detach().to(self.device)
        
#         output_neurons = tensor.shape[1]
#         neuron_score = torch.zeros((output_neurons)).to(device = self.device)
        
#         if self.l1_norm:
#             neuron_score += self.normalize_scores(self.l1_norm_score(tensor, dim = 0))
            
#         if self.l2_norm: 
#             neuron_score += self.normalize_scores(self.l2_norm_score(tensor, dim = 0))
                
#         return neuron_score
    
    
#     def get_head_score(self, tensor, num_heads, head_dim):
        
#         tensor = tensor.clone().detach().to(self.device)
        
#         assert len(tensor.shape) == 2
        
#         input_neurons, output_neurons = tensor.shape
#         tensor = tensor.view(input_neurons, num_heads, head_dim).transpose(0, 1).contiguous()
        
#         head_score = torch.zeros((num_heads)).to(device = self.device)

#         for idx, head in enumerate(tensor):
#             head_score[idx] = torch.norm(head, p = 2)

#         head_score = self.normalize_scores((head_score))
            
#         del tensor
        
#         return head_score
    


#     def get_head_score_activation(self, tensor, num_heads, head_dim):
        
#         tensor = tensor.clone().detach().to(self.device)
#         assert len(tensor.shape) == 3
#         assert list(tensor.shape) == [num_heads, 2048, head_dim]

#         # tensor = tensor.view(input_neurons, num_heads, head_dim).transpose(0, 1).contiguous()
        
#         head_score = torch.zeros((num_heads)).to(device = self.device)

#         for idx, head in enumerate(tensor):
#             head_score[idx] = torch.norm(head, p = 2)

#         head_score = self.normalize_scores((head_score))
            
#         del tensor
        
#         return head_score

    
    
#     def store_attn_weights(self, weight, num_heads, head_dim):
        
#         self.weights_updated = True
        
#         self.weight_tensor = weight.clone().detach().to(self.device)
#         self.weight_tensor = torch.transpose(self.weight_tensor, 0, 1)
        
#         self.weight_score_head = self.get_head_score(self.weight_tensor.clone().detach(), num_heads, head_dim)
       
    
#     def store_FFN_weights(self, weight):
        
#         self.weights_updated = True
        
#         self.weight_tensor = weight.clone().detach().to(self.device)
#         self.weight_tensor = torch.transpose(self.weight_tensor, 0, 1)
        
#         self.weight_score_neuron = self.get_neuron_score(self.weight_tensor.clone().detach())


#     def store_gradients(self, grad):
        
#         grad = grad.clone().detach().t().to(device = self.device)
        
#         if self.grad_updated:
#             self.grad_avg = (self.grad_avg * self.activation_samples + grad) / (self.activation_samples + 1)
#         else:
#             self.grad_avg = grad
#         self.grad_updated = True
        
        
#     def get_attn_gradients_score(self, num_heads, head_dim):
        
#         if not self.grad_updated:
#             if not self.weights_updated:
#                 print("Gradients are not updated. Exiting")
#                 exit()
            
#         grad_score_head = self.get_head_score(self.grad_avg, num_heads, head_dim)
#         return grad_score_head 
    
    
#     def get_FFN_gradients_score(self):
        
#         if not self.grad_updated:
#             if not self.weights_updated:
#                 print("Gradients are not updated. Exiting")
#                 exit()
            
#         grad_score_neuron = self.get_neuron_score(self.grad_avg)
#         return grad_score_neuron 
    
    
        
#     def store_activation(self, act):
        
#         act = act.clone().to(device = self.device)
#         act = act.clone().detach().squeeze(0)
        
#         if self.acts_updated:
#             self.act_avg = (self.act_avg * self.activation_samples + act) / (self.activation_samples + 1)
            
#         else:
#             self.act_avg = act
        
#         self.acts_updated = True






    
#     def get_attn_activation_score(self, num_heads, head_dim):
        
#         if not self.acts_updated:
#             print("Activations not updated. Exiting")
#             exit()
            
#         activation_score_head = self.get_head_score_activation(self.act_avg, num_heads, head_dim)
        
#         return activation_score_head
    
    
#     def get_head_act_score(self, num_heads, head_dim):
#         activation_score_head = self.get_attn_activation_score(num_heads, head_dim)
#         return activation_score_head 
    
    
#     def get_FFN_activation_score(self):
        
#         if not self.acts_updated:
#             print("Activations not updated. Exiting")
#             exit()
            
#         activation_score_neuron    = self.get_neuron_score(self.act_avg)
        
#         return activation_score_neuron
    
    
    
#     def get_FFN_layer_score(self):
        
#         score_neuron = 0
        
#         if self.FFN_act_score:
#             assert self.acts_updated == True
#             activation_score_neuron = self.get_FFN_activation_score()

#             if self.FFN_wt_score:
#                 assert self.weight_score_neuron.shape == activation_score_neuron.shape    
            
#             score_neuron += activation_score_neuron


#         if self.wt_mult_grad_plus_act:
#             assert self.weights_grad_updated == True
#             score_neuron += self.weight_grad_FFN_score()

#         else:
#             assert self.wt_plus_grad_plus_act == True

#             if self.FFN_wt_score:
#                 assert self.weights_updated == True
#                 score_neuron += self.weight_score_neuron

#             if self.FFN_grad_score:
#                 assert self.grad_updated == True
#                 grad_score_neuron = self.get_FFN_gradients_score()

#                 if self.FFN_wt_score:
#                     assert self.weight_score_neuron.shape == grad_score_neuron.shape
#                 score_neuron += grad_score_neuron
                
#         return score_neuron
    
    
    
#     def get_attention_layer_score(self, num_heads, head_dim):
        
#         score_head = 0
        
#         if self.Attention_act_score:
#             assert self.acts_updated == True
#             activation_score_head = self.get_attn_activation_score(num_heads, head_dim)

#             if self.Attention_wt_score:
#                 assert self.weight_score_head.shape == activation_score_head.shape
#             score_head += activation_score_head

#         if self.wt_mult_grad_plus_act:
#             assert self.weights_grad_updated == True
#             score_head += self.get_weight_grad_attn_score(num_heads, head_dim)


#         else:
#             if self.Attention_wt_score:
#                 assert self.weights_updated == True
#                 score_head += self.weight_score_head 
                
#             if self.Attention_grad_score:
#                 assert self.grad_updated == True
#                 grad_score_head = self.get_attn_gradients_score(num_heads, head_dim)

#                 if self.Attention_wt_score:
#                     assert self.weight_score_head.shape == grad_score_head.shape
#                 score_head += grad_score_head
                
#         return score_head
    
    
    
#     def get_layer_prune_mask(self, sparsity, prune_dim_nearest, attention, num_heads, head_dim):
#         layer_score   =  self.get_layer_score(attention, num_heads, head_dim)
#         prune_nodes   =  np.round((self.prune_dimension*sparsity)/(prune_dim_nearest*100))*prune_dim_nearest 
#         thresh        =  torch.sort(layer_score.flatten())[0][int(prune_nodes)]
#         mask          =  layer_score >= thresh

#         # print("score", score)
#         # print("thresh", thresh)

#         print("prune_dimension", self.prune_dimension)
#         print("sparsity", sparsity)
#         print("prune_dim_nearest", prune_dim_nearest)
#         print("prune_nodes", prune_nodes)
#         print("bool mask", sum(mask))
#         print("True Nodes", self.prune_dimension-prune_nodes)

#         exit()

#         return mask
        
    
    
    # def l1_norm_score(self, tensor, dim):
    #     tensor = tensor.clone().float()
    #     l1_score = torch.sum(torch.abs(tensor), dim = dim)
    #     return l1_score
    
    
    # def l2_norm_score(self, tensor, dim):
    #     tensor = tensor.clone().float()
    #     l2_score = torch.sqrt(torch.sum(torch.pow(tensor, 2), dim = dim))
    #     return l2_score
    
    
    
    
    
if __name__ == '__main__':
    
    hidden_size = 4096
    inner_hidden_size = 2048
    num_heads = 32
    head_dim = inner_hidden_size // num_heads
    num_key_value_heads  = 32
    num_key_value_groups = num_heads // num_key_value_heads
    max_position_embeddings = 2048

    q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
    v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    
    prune_API_MHSA_q    = PruneGPT(prune_dimension = num_heads)
    prune_API_MHSA_k    = PruneGPT(prune_dimension = num_heads)
    prune_API_MHSA_v    = PruneGPT(prune_dimension = num_heads)
    prune_API_MHSA_attn = PruneGPT(prune_dimension = num_heads)
    prune_API_MHSA_o    = PruneGPT(prune_dimension = num_heads)

    
    prune_API_MHSA_q.save_device(device    = q_proj.weight.device)   
    prune_API_MHSA_k.save_device(device    = k_proj.weight.device)   
    prune_API_MHSA_v.save_device(device    = v_proj.weight.device)   
    prune_API_MHSA_attn.save_device(device = q_proj.weight.device)
    prune_API_MHSA_o.save_device(device    = o_proj.weight.device)   

    prune_API_MHSA_q.set_pruning_choices(None)
    prune_API_MHSA_k.set_pruning_choices(None)
    prune_API_MHSA_v.set_pruning_choices(None)
    prune_API_MHSA_attn.set_pruning_choices(None)
    prune_API_MHSA_o.set_pruning_choices(None)
    
    
    arg_dict = {"attention" : True,
                "hidden_size": hidden_size,
                "inner_hidden_size": inner_hidden_size,
                "num_heads": num_heads,
                "head_dim": head_dim}
        
    
    prune_API_MHSA_q.update_weights(q_proj.weight, **arg_dict)
    prune_API_MHSA_k.update_weights(k_proj.weight, **arg_dict)
    prune_API_MHSA_v.update_weights(v_proj.weight, **arg_dict)
    
    query_states   = torch.rand((1, max_position_embeddings, inner_hidden_size))
    key_states     = torch.rand((1, max_position_embeddings, inner_hidden_size))
    value_states   = torch.rand((1, max_position_embeddings, inner_hidden_size))
    attn_output_1  = torch.rand((1, max_position_embeddings, inner_hidden_size))
    attn_output_2  = torch.rand((1, max_position_embeddings, inner_hidden_size))
    
    
    prune_API_MHSA_q.update_activation(query_states)
    prune_API_MHSA_k.update_activation(key_states)
    prune_API_MHSA_v.update_activation(value_states)
    prune_API_MHSA_attn.update_activation(attn_output_1)
    prune_API_MHSA_o.update_activation(attn_output_2)
    
    
    q_proj_weight_grad = torch.rand((inner_hidden_size, hidden_size))
    k_proj_weight_grad = torch.rand((inner_hidden_size, hidden_size))
    v_proj_weight_grad = torch.rand((inner_hidden_size, hidden_size))
    o_proj_weight_grad = torch.rand((inner_hidden_size, hidden_size))
    
    
    prune_API_MHSA_q.store_gradients(q_proj_weight_grad) 
    prune_API_MHSA_k.store_gradients(k_proj_weight_grad) 
    prune_API_MHSA_v.store_gradients(v_proj_weight_grad)
    prune_API_MHSA_o.store_gradients(o_proj_weight_grad)
    
    
    prune_API_MHSA_q.update_activation_samples()
    prune_API_MHSA_k.update_activation_samples()
    prune_API_MHSA_v.update_activation_samples()
    prune_API_MHSA_attn.update_activation_samples()
    prune_API_MHSA_o.update_activation_samples()

    
    dict_args = {"attention":True, "num_heads": num_heads, "head_dim": head_dim}
            
    qkv_score =  get_qkv_avg(prune_API_MHSA_q.get_layer_score(**dict_args),
                             prune_API_MHSA_k.get_layer_score(**dict_args),
                             prune_API_MHSA_v.get_layer_score(**dict_args)
                             )

    qkv_score = qkv_score + prune_API_MHSA_attn.get_activation_score(**dict_args)

    prune_mask_qkv = get_prune_mask_from_score(prune_dimension = num_heads,
                                               score = qkv_score,
                                               sparsity = 0.05, 
                                               prune_dim_nearest = 1)

    dict_args["prune_mask"] = prune_mask_qkv
    
    q_proj     = prune_output_linear_layer(q_proj, **dict_args)
    k_proj     = prune_output_linear_layer(k_proj, **dict_args)
    v_proj     = prune_output_linear_layer(v_proj, **dict_args)
    o_proj     = prune_input_linear_layer(o_proj,  **dict_args)

    del prune_mask_qkv

    assert q_proj.weight.shape[0] % head_dim == 0
    num_heads = int(q_proj.weight.shape[0]/head_dim)
    num_key_value_heads = int(k_proj.weight.shape[0]/head_dim)
    
    
    
