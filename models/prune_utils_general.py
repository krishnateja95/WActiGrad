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
        
        if args.wt_plus_grad_plus_act:
            model.store_grads()
        else:
            model.store_weight_grad()


        model.update_activation_samples()

    if args.wt_plus_grad_plus_act:
        model.store_weights()

    torch.cuda.empty_cache()
    
    model.prune_fc_layer()
    model.set_calibration(calibration = False)    
    
    _ = model(inputs).logits

    print("End of Calibration")

    return model 


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
    

def l1_norm_score(tensor, dim):
    tensor = tensor.clone().float()
    l1_score = torch.sum(torch.abs(tensor), dim = dim)
    return l1_score


def l2_norm_score(tensor, dim):
    tensor = tensor.clone().float()
    l2_score = torch.sqrt(torch.sum(torch.pow(tensor, 2), dim = dim))
    return l2_score


def normalize_scores(score, normalizer):
    if normalizer == "softmax":
        return F.softmax(score, dim = -1)
    elif normalizer == "sum":
        return score / score.sum()
    elif normalizer == "standarization":
        return (score - score.min()) / (score.max() - score.min()+1e-8)
    elif normalizer == "mean":
        return score / score.mean()
    elif normalizer == "max":
        return score / score.max()
    elif normalizer == 'gaussian':
        return (score - score.mean()) / (score.std()+1e-8)
    else:
        raise NotImplementedError






