import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(tokenizer):
    
    train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    test_dataset  = load_dataset('wikitext',  'wikitext-2-raw-v1', split='test')
    
    train_data_encoded = train_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
    test_data_encoded  = test_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

    return train_data_encoded, test_data_encoded

def get_ptb(tokenizer):
    
    train_dataset = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    test_dataset  = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    
    train_data_encoded = train_dataset.map(lambda samples: tokenizer(samples['sentence']), batched=True)
    test_data_encoded  = test_dataset.map(lambda samples: tokenizer(samples['sentence']), batched=True)
    
    return train_data_encoded, test_data_encoded


def get_c4(tokenizer):
    
    train_dataset = load_dataset('allenai/c4', 'allenai--c4', 
                             data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    
    test_dataset  = load_dataset('allenai/c4', 'allenai--c4', 
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    train_data_encoded = train_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
    test_data_encoded  = test_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
    return train_data_encoded, test_data_encoded



def get_loaders(name, tokenizer=None):
    
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer)
    
    if 'ptb' in name:
        return get_ptb(tokenizer)
    
    if 'c4' in name:
        return get_c4(tokenizer)
    
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    _,_ = get_loaders(name = 'c4',        tokenizer=tokenizer)
    _,_ = get_loaders(name = 'ptb',       tokenizer=tokenizer)
    _,_ = get_loaders(name = 'wikitext2', tokenizer=tokenizer)
    
    
        
    
    
    
    
    
    
    