

conda activate wactigrad

model="meta-llama/Llama-2-7b-hf"
dataset="ptb"

act_flags=" --Attention_wt_score --Attention_grad_score --Attention_act_score --FFN_wt_score --FFN_grad_score --FFN_act_score"
gran_flags_1=" --l2_norm --prune_MLP --prune_hidden_size --prune_Head"

for num_head in 0 1 2 3; do
  for hidden_size_sparsity in 0 5 10 15 20 25; do
    for MLP_sparsity in 0 5 10 15 20 25; do
      python wactigrad.py --model=$model $act_flags --dataset=$dataset $gran_flags_1 --wt_mult_grad_plus_act --prune_Heads_num=$num_head --prune_MLP_sparsity=$MLP_sparsity --prune_hidden_size_sparsity=$hidden_size_sparsity
    done
  done 
done 

