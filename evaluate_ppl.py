import time
import torch
import torch.nn as nn

from data import get_loaders 

def eval_ppl(model, tokenizer, dataset, device = torch.device("cuda:0")):
    ppl_dict = {}
    trainloader, testloader = get_loaders(dataset, seed=0, nsamples=128, seqlen=model.seqlen, tokenizer=tokenizer)
    with torch.no_grad():
        print("Evaluating done on ", dataset)
        ppl_dict[dataset + "_ppl_test"] = eval_ppl_test(model, testloader, 1, device) 
        ppl_dict[dataset + "_ppl_train"] = eval_ppl_train(model, trainloader, 1, device)
    return ppl_dict


def eval_ppl_train(model, trainloader, bs = 1, device = None):
    nsamples = len(trainloader)
    
    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)

        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    torch.cuda.empty_cache()

    return ppl.item()

def eval_ppl_test(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids

    nsamples = testenc.numel() // model.seqlen
    
    nlls = []
    print(f"nsamples {nsamples}")
    
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()
    return ppl.item()