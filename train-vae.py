from utils.data_reader import Tasks
from model.vae import VAE
from model.common_layer import evaluate_vae as evaluate
from utils import config
import torch
#torch.cuda.set_device(2)
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time 
import numpy as np 
import pickle

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

p = Tasks()

data_loader_tr, data_loader_val, data_loader_test = p.get_all_data(batch_size=config.batch_size)

if(config.test):
    print("Test model",config.model)
    model = VAE(p.vocab,model_file_path=config.save_path,is_eval=True)
    evaluate(model,data_loader_test,model_name=config.model,ty='test',verbose=True,log=True)
    exit(0)

model = VAE(p.vocab)
print("MODEL USED",config.model)
print("TRAINABLE PARAMETERS",count_parameters(model))

best_ppl = 1000
cnt = 0
for e in range(config.epochs):
    print("Epoch", e)
    p, l = [],[]
    pbar = tqdm(enumerate(data_loader_tr),total=len(data_loader_tr))
    for i, d in pbar:
        torch.cuda.empty_cache()
        loss, ppl, total_loss, re_loss, kl_loss, bow_loss = model.train_one_batch(d)
        l.append(loss)
        p.append(ppl)
        #pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),np.mean(p)))
        pbar.set_description("loss:{:.4f} ppl:{:.1f} total_loss:{:.4f} re_loss:{:.4f} kl_loss:{:.4f} bow_loss:{:.4f}".format(loss,ppl,total_loss, re_loss, kl_loss, bow_loss))
        torch.cuda.empty_cache()
        
    loss,ppl_val,ent_b,bleu_score_b = evaluate(model,data_loader_val,model_name=config.model,ty="valid", verbose=True)
    if(ppl_val <= best_ppl):
        best_ppl = ppl_val
        cnt = 0
        model.save_model(best_ppl,e,0,0,bleu_score_b,ent_b)
    else: 
        cnt += 1
    if(cnt > 10): break
 



