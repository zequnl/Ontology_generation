from utils.data_reader import Data
from model.vae_graph import VAE
from model.common_layer import evaluate_graph as evaluate
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
file_list = os.listdir("data/")
ppls = {}
bleus = {}
for di in file_list:
    if di == ".ipynb_checkpoints":
        continue
    #di = "gaz"
    if di != "go":
        continue
    config.data_dir = "data/" + di
    config.save_path = "save_graph1/" + di + "/"
    config.save_path_dataset = "save_graph1/" + di + "/"
    p = Data()
    print(di)

    data_loader_all, data_loader_tr, data_loader_val, data_loader_test = p.get_all_data(batch_size=config.batch_size)
    for i,d in enumerate(data_loader_all):
        all_data = d
    #print(data_loader_all[0])
    model = VAE(p.vocab, p.graph, 1)
    print("MODEL USED",config.model)
    print("TRAINABLE PARAMETERS",count_parameters(model))

    best_ppl = 1000
    cnt = 0
    best_model = model.state_dict()
    for e in range(config.epochs):
        print("Epoch", e)
        p, l = [],[]
        pbar = tqdm(enumerate(data_loader_tr),total=len(data_loader_tr))
        for i, d in pbar:
            torch.cuda.empty_cache()
            #for i, dt in enumerate(data_loader_all):
                #model.get_graph_feature(dt)
            loss, ppl, total_loss, re_loss, kl_loss, bow_loss = model.train_one_batch(d, all_data)
            l.append(loss)
            p.append(ppl)
            pbar.set_description("loss:{:.4f} ppl:{:.1f} total_loss:{:.4f} re_loss:{:.4f} kl_loss:{:.4f} bow_loss:{:.4f}".format(loss,ppl,total_loss, re_loss, kl_loss, bow_loss))
        #pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(loss,ppl))
            torch.cuda.empty_cache()
            #if i > 1:
                #break
        #break
        #for i, d in enumerate(data_loader_total):
            #model.get_graph_feature(d)
        #break
        loss,ppl_val,ent_b,bleu_score_b = evaluate(model,data_loader_val,all_data, model_name=config.model,ty="valid", verbose=False)
        if(ppl_val <= best_ppl):
            best_ppl = ppl_val
            cnt = 0
            best_model = model.state_dict()
            model.save_model(best_ppl,e,0,0,bleu_score_b,ent_b)
        else: 
            cnt += 1
        if(cnt > 10): 
            break
    #break
    model.load_state_dict(best_model)
    if not os.path.exists("results2/" + di + "/"):
        os.makedirs("results2/" + di + "/")
    loss,ppl,ent_b,bleu_score_b = evaluate(model,data_loader_test,all_data, model_name=config.model,ty='test',verbose=False,log=True, result_file="results2/" + di + "/results_vae_graph1.txt", ref_file="results2/" + di + "/ref_vae_graph1.txt", case_file="results2/" + di + "/case_vae_graph1.txt")
    ppls[di] = ppl
    bleus[di] = bleu_score_b
    #break
    
    
avg_ppl = 0
avg_bleu = 0
for d in ppls:
    print(d, ppls[d], bleus[d])
    avg_ppl += ppls[d]
    avg_bleu += bleus[d]
avg_ppl /= len(file_list)
avg_bleu /= len(file_list)
print(avg_ppl, avg_bleu)

    
            
 



