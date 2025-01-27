# python Sparse_CasualLM_fine_tune_PEFT_LORA_LLaMA.py --wt_score --act_score --attn_score --grad_score --l1_norm --l2_norm --prune_MHSA --prune_MLP --prune_hidden_size

from peft import LoraConfig, get_peft_model
import os
import time
import torch
import csv
import torch.nn as nn
import transformers
from transformers.trainer import Trainer
import argparse

from models.get_model import get_LLM
from models.prune_utils_general import prune_moe
from train_dataloaders import get_loaders
from evaluate_ppl import eval_ppl


def calculate_pruning_percentage(original_params, pruned_params):
    num_parameters_pruned = original_params - pruned_params
    pruning_percentage = (num_parameters_pruned / original_params) * 100
    return pruning_percentage


def count_non_zero_params(model):
    non_zero_params = 0
    total_params    = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            non_zero_params += torch.count_nonzero(param).item()
            total_params += param.numel()

    return non_zero_params, total_params


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model',      type=str,   default='mistralai/Mistral-7B-v0.1', help='LLM model')
    parser.add_argument('--seed',       type=int,   default=0,    help='Seed for sampling the calibration data.')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps for fine tuning')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Max steps for fine tuning')
    parser.add_argument('--save',       type=str,   default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str,   default=None, help='Path to save the pruned model.')
    parser.add_argument('--dataset',  type=str, default="ptb", help='Calibration Dataset')
    parser.add_argument('--nsamples_calibration', type=int, default=20, help='Calibration Dataset')
    parser.add_argument('--batch_size',   type=int, default=1,  help='Batch Size')
    parser.add_argument('--num_gpus',   type=int, default=2,  help='Number of GPUs')

    parser.add_argument("--Attention_wt_score",     action = "store_true",  help = "Weight Score in pruning mask")
    parser.add_argument("--Attention_act_score",    action = "store_true",  help = "Activation Score in pruning mask")
    parser.add_argument("--Attention_grad_score",   action = "store_true",  help = "Gradient Score in pruning mask")
    
    parser.add_argument("--FFN_wt_score",   action = "store_true",  help = "Attention  Score in pruning mask")
    parser.add_argument("--FFN_act_score",   action = "store_true",  help = "Gradient Score in pruning mask")
    parser.add_argument("--FFN_grad_score",   action = "store_true",  help = "Attention  Score in pruning mask")
    
    parser.add_argument("--l1_norm",      action = "store_true", help = "Weight Score in pruning mask")
    parser.add_argument("--l2_norm",      action = "store_true", help = "Weight Score in pruning mask")
    
    parser.add_argument("--prune_Head",         action = "store_true", help = "Weight Score in pruning mask")
    parser.add_argument("--prune_MLP",          action = "store_true", help = "Weight Score in pruning mask")
    parser.add_argument("--prune_hidden_size",  action = "store_true", help = "Weight Score in pruning mask")
    
    parser.add_argument("--prune_Heads_num",             type=int, default=2, help = "Weight Score in pruning mask")
    parser.add_argument("--prune_MLP_sparsity",          type=int, default=5, help = "Weight Score in pruning mask")
    parser.add_argument("--prune_hidden_size_sparsity",  type=int, default=5, help = "Weight Score in pruning mask")

    parser.add_argument("--per_device_train_batch_size",  type=int, default=1, help = "Weight Score in pruning mask")
    parser.add_argument("--gradient_accumulation_steps",  type=int, default=4, help = "Weight Score in pruning mask")
    
    parser.add_argument("--normalize",  type=str, default = "mean", help = "Weight Score in pruning mask")
    
    parser.add_argument("--wt_plus_grad_plus_act",  action = "store_true", help = "Weight Score in pruning mask")
    parser.add_argument("--wt_mult_grad_plus_act",  action = "store_true", help = "Weight Score in pruning mask")
    
    args = parser.parse_args()
    
    model, tokenizer = get_LLM(args.model, int(torch.cuda.device_count()))

    _, total_params = count_non_zero_params(model)
    
    device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    model  = prune_moe(args,
                       model,
                       tokenizer,
                       device,
                       nsamples_calibration = args.nsamples_calibration,
                       bs = args.batch_size, 
                       calibration_dataset = args.dataset)
    
    non_zero_params, total_pruned_params = count_non_zero_params(model)
    
    print(f"Number of non-zero parameters: {non_zero_params}")
    print(f"Total number of parameters: {total_params}")
    
    print(model)
    
    model.config.use_cache = True
    
    device = model.model.embed_tokens.weight.device
    
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    
    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    config = LoraConfig(r              = 8, 
                        lora_alpha     = 16,
                        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
                        lora_dropout   = 0.05,
                        bias           = "none",
                        task_type      = "CAUSAL_LM"
                       )
    
    model = get_peft_model(model, config)
    
    train_data_encoded, test_data_encoded = get_loaders(args.dataset, tokenizer = tokenizer)
    
    print_trainable_parameters(model)
    
    model.config.use_cache = True
    
    trainer = Trainer(model = model,
                      train_dataset = train_data_encoded,
                      eval_dataset  = test_data_encoded,
                      tokenizer     = tokenizer,
                      args=transformers.TrainingArguments(per_device_train_batch_size=1,
                                                          gradient_accumulation_steps=4,
                                                          warmup_steps=args.warmup_steps,
                                                          max_steps=args.max_steps,
                                                          learning_rate=2e-4,
                                                          fp16=True,
                                                          logging_steps=25,
                                                          output_dir="./LLaMA_output_dir"),
                      data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
    
    start_time    = time.time()
    train_results = trainer.train()
    end_time      = time.time()
    
    finetuning_time = end_time - start_time
    
    finetune_dict = eval_ppl(model, tokenizer, args.dataset, device)
    
    def split_string(model_name):
        if "/" in model_name:
            return model_name.split("/")[-1]
        else:
            return model_name   

    pruning_percentage = calculate_pruning_percentage(total_params, total_pruned_params)

    list_1 = ["Model Name",
              "prune_Heads_num",
              "prune_MLP_sparsity",
              "prune_hidden_size_sparsity",
              "pruning_percentage",
              "Prune Hidden Size",
              "Prune MLP",
              "Prune Head", 
              "wt_plus_grad_plus_act",
              "wt_mult_grad_plus_act",
              "L2-norm",
              "Attention_wt_score",
              "Attention_act_score",
              "Attention_grad_score",
              "FFN_wt_score",
              "FFN_act_score",
              "FFN_grad_score",
              "Normalize",
              "per_device_train_batch_size",
              "gradient_accumulation_steps",
              "finetuning_time",
              "dataset",
              "train_samples_per_second"
              ]
        

    list_2 = [split_string(args.model),
              args.prune_Heads_num,
              args.prune_MLP_sparsity,
              args.prune_hidden_size_sparsity,
              pruning_percentage,
              args.prune_hidden_size,
              args.prune_MLP,
              args.prune_Head,
              args.wt_plus_grad_plus_act,
              args.wt_mult_grad_plus_act,
              args.l2_norm,
              args.Attention_wt_score,
              args.Attention_act_score,
              args.Attention_grad_score,
              args.FFN_wt_score,
              args.FFN_act_score,
              args.FFN_grad_score,
              args.normalize,
              1,
              4,
              finetuning_time,
              args.dataset,
              train_results[2]["train_samples_per_second"]
              ] 


    ppl_dataset_phase = list_1 + [key for key in finetune_dict]
    ppl_values = list_2 + [v for v in finetune_dict.values()]
    
    assert len(ppl_dataset_phase) == len(ppl_values)

    with open("Results.csv", 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ppl_dataset_phase)
        writer.writerow(ppl_values) 
        
    csvfile.close()























