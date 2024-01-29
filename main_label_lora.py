from transformers import (
    AutoConfig,
    AutoModel,
    BertTokenizer,
)
from model_2 import MutilDefendantCrimePre
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np

import torch
from dataProcess import criminal_sample
from tqdm import tqdm
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from Data import load_data
from utlil import onelabel_acc_fn,logging,ensureDirs,multilabel_acc_fn,labels2one_hot,calculate_f1_and_recall,get_F1_recall_logging,get_logging,get_num_trainable_parameters
import datetime
import json
from models import *
from peft import LoraConfig
from torch.utils.data import DataLoader

def eval(model,eval_dataloader,device,loss_fn,type,charge_def_emb):
    total_f1 = []
    total_recall = []
    total_loss = []
    model.eval()
    for item in tqdm(eval_dataloader):
        token_len = item[0].size()[-1]
        input_ids = item[0].view(-1, token_len).to(device)
        attention_mask = item[1].view(-1, token_len).to(device)
        seg = item[2].view(-1, token_len).to(device)
        if type == 3:
            labels = item[3].to(device)
        else :
            labels = labels2one_hot(item[3], num_classes=cus_config['num_labels'])
            labels = labels.view(-1, labels.size()[-1]).to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids = seg,charge_def = charge_def_emb)
        loss = loss_fn(output, labels)
        f1,recall = calculate_f1_and_recall(labels,output,threshold=0.5)
        total_f1.append(f1)
        total_recall.append(recall)
        total_loss.append(loss.data.cpu().numpy())
    return np.array(total_f1).mean(),np.array(total_recall).mean(),np.array(total_loss).mean()

def eval_1(model,eval_dataloader,device,loss_fn = None):
    total_acc = []
    total_loss = []
    model.eval()
    for item in tqdm(eval_dataloader):
        token_len = item[0].size()[-1]
        input_ids = item[0].view(-1, token_len).to(device)
        attention_mask = item[1].view(-1, token_len).to(device)
        seg = item[2].view(-1, token_len).to(device)

        labels = torch.tensor(item[3],dtype=torch.float32).to(device).unsqueeze(dim = 1)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=seg)
        loss = loss_fn(output, labels)

        total_acc.append(multilabel_acc_fn(labels,output).data.cpu().numpy())
        total_loss.append(loss.data.cpu().numpy())

    return  np.array(total_acc).mean(), np.array(total_loss).mean()



config = {
    #train config
    'epoch':30,
    'num_optimizer_step':None,
    'warm_up_proportion':0.1,
    'lr_base':2e-5,
    'seed':42,
    'batch_size':2,
    'gradient_accumulation_steps':8,

    # type 1: multi q multi defendant
    # type 2:isOrNot Crime
    # type 3:global pre for entire fact
    # type 4:multi_q transformer
    "type":1,


    #log and save model path
    'log_dir':'./log',
    'log_file_name':'r_512_lora_label.txt',
    'save_model':'./fusion_charge_result',
    'model_name':'r_512.pt',
    'early_stop':False,
    'patience':5
}

cus_config = {
    "num_labels":1 if config['type'] == 2 else 72,
    "hidden_dropout_prob":0.5,
    "hidden_size":768,
    'graph_num':2,
    "MultiLabel":True,
    "charge_fusion":True,
    "label_encoder_path":"/root/autodl-tmp/BMRC-main/bert-base-uncased",
    "peft_config":PLoraConfig(task_type="SEQ_CLS" ,
                                  inference_mode=False,
                                  r=512,
                                  lora_alpha=128,
                                  lora_dropout=0.1,
                                #target_modules=['query', 'key', 'value', 'intermediate.dense', 'output.dense'],
                                #target_modules=['value', 'key',"key_global","value_global"],
                                  user_token_dim=768),
    #"peft_config":None,
    #"peft_config":LoraConfig( inference_mode=False, r=64, lora_alpha=128, lora_dropout=0.1)
}

data_dir = './mutil_defendant_mutil_crime/new_processed'
train_data_dir = './mutil_defendant_mutil_crime/new_processed/single_style_train'
#loading charge def
charge_def_list = []
charge_path = './mutil_defendant_mutil_crime/charge_def_new.json'
with open(charge_path,'r',encoding='utf-8') as f:
    charge_def = json.load(f)

for k,v in charge_def.items():
    charge_def_list.append(v)

#model_path = '../longformer_zh'
model_path = "../longformer_zh"
##data load
torch.manual_seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['seed'])

tokenizer = BertTokenizer.from_pretrained(model_path,trust_remote_code=True)
_,val_dataloader,test_dataloader = load_data(data_dir,batch_size=config['batch_size'],tokenizer=tokenizer,type=config['type'])
from Data import single_style_train_Dataset
train_set = []
for i in range(len(os.listdir(train_data_dir))):
    data_item = torch.load(f'./mutil_defendant_mutil_crime/new_processed/single_style_train/data_{i}.pt')
    train_set.append(data_item)
train_data = single_style_train_Dataset(train_set)
train_dataloader = DataLoader(train_data,batch_size=1,shuffle=True)

config['num_optimizer_step'] = len(train_dataloader)*config['epoch']/config['gradient_accumulation_steps']

charge_def = tokenizer(charge_def_list,padding=True,truncation=True,return_tensors='pt')
##load model

model = MutilDefendantCrimePre(pretrain_path=model_path, cus_config=cus_config)
num_parameter = get_num_trainable_parameters(model)
device = torch.cuda.current_device()
model = model.to(device)
charge_def = charge_def.to(device)



##load optimizer and scheduler and loss_fn
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr_base'], weight_decay=0.01, correct_bias=False)
scheduler = WarmupLinearSchedule(optimizer,num_training_steps=config['num_optimizer_step'],num_warmup_steps=config['num_optimizer_step']*config['warm_up_proportion'])
loss_fn = nn.CrossEntropyLoss()
acc_fn = multilabel_acc_fn
if cus_config["MultiLabel"]:
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    acc_fn = calculate_f1_and_recall
ensureDirs(config['log_dir'])

logfile = open(os.path.join(config['log_dir'],config['log_file_name']),'a+')
logfile.write(
            '\n' +
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(config['seed'])
            )
logfile.close()

best_dev_f1 = 0
best_dev_acc = 0
unimproved_iters = 0
gradient_accumulation_count = 1
count=0
count_step = 1
s = 0
global_step = 0
scaler = GradScaler()
model.train()
for i in range(config['epoch']):
    total_loss = []
    total_F1 = []
    total_recall = []
    total_acc = []

    for item in tqdm(train_dataloader):
        # if s < 3:
        #     s+=1
        #     continue
        global_step+=1


        input_ids = item["input_ids"].to(device)
        attention_mask = item["attention_mask"].to(device)
        seg = item["token_type_ids"].to(device)

        if config['type'] == 2:
            labels = torch.tensor(item[3],dtype=torch.float32).to(device).unsqueeze(dim = 1)
        elif config['type'] == 3:
            labels = item[3].to(device)
        else:
            labels = item["label"].to(device)


        with autocast():
            output = model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids = seg,charge_def = charge_def)

            if config['type'] == 2 or config['type'] == 3:
                loss = loss_fn(output,labels)
            else:
                loss = loss_fn(output,labels)

        if torch.isnan(loss).any():
            print("The num of {} loss is NaN".format(count_step))
            continue
        if config['type'] == 2:
            total_acc.append(multilabel_acc_fn(labels,output).data.cpu().numpy())
        else:
            f1,recall = calculate_f1_and_recall(y_true=labels,y_pred=output,threshold=0.5)
            total_F1.append(f1)
            total_recall.append(recall)


        if config['gradient_accumulation_steps']>1:
            loss = loss/config['gradient_accumulation_steps']
        scaler.scale(loss).backward()
        # loss.backward()
        if gradient_accumulation_count==config['gradient_accumulation_steps']:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            gradient_accumulation_count=1
            count+=1
        else:
            gradient_accumulation_count+=1
        total_loss.append(loss.data.cpu().numpy())
        if global_step % 100 == 0:
            print("Training loss: {}, global step: {}".format(np.array(total_loss).mean(), global_step))

    if config['type'] == 2:
        logs = ("    Epoch:{:>2}    ".format(i)).center(88, "-") + "".center(70, " ") + '\n' + \
           get_logging(np.array(total_loss).mean(),np.array(total_acc).mean(),eval="training")
    else:
        logs = ("    Epoch:{:>2}    ".format(i)).center(88, "-") + "".center(70, " ") + '\n' + \
            get_F1_recall_logging(np.array(total_F1).mean(),np.array(total_recall).mean(),np.array(total_loss).mean(),eval="training")
    print("\r" + logs)

    # logging training logs
    logging(log_file=os.path.join(config['log_dir'],config['log_file_name']),logs=logs)

    if config['type'] == 2:
        eval_acc,eval_loss = eval_1(model,val_dataloader,device=device,loss_fn=loss_fn)
        eval_logs = get_logging(eval_loss, eval_acc, eval="evaluating")
    else:
        eval_f1,eval_recall,eval_loss = eval(model,val_dataloader,device=device,loss_fn=loss_fn,type = config['type'],charge_def_emb=charge_def)
        eval_logs = get_F1_recall_logging(eval_f1,eval_recall,eval_loss, eval="evaluating")
    print("\r" + eval_logs)
    logging(os.path.join(config['log_dir'],config['log_file_name']),eval_logs)

    if eval_f1>best_dev_acc:
        unimproved_iters=0
        best_dev_acc = eval_f1
        best_state_dict = model.state_dict()
        # saving models
        ensureDirs(config['save_model'])
        torch.save(best_state_dict, os.path.join(config['save_model'], config['model_name']))
    else:
        unimproved_iters += 1
        if unimproved_iters >= config['patience'] and config['early_stop'] == True:
            early_stop_logs = "Early Stopping. Epoch: {}, Best Dev F1: {}".format(i, best_dev_f1)
            print(early_stop_logs)
            logging(log_file=os.path.join(config['log_dir'],config['log_file_name']),logs=early_stop_logs)
            break