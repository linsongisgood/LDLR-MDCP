from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer,AutoTokenizer
import numpy as np
from utlil import labels2one_hot
from dataProcess import criminal_sample
from torch.nn.utils.rnn import pad_sequence

tokenizer = BertTokenizer.from_pretrained('../longformer_zh')
#tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/bert-chinese')
class Mutil_def_Dataset(Dataset):
    def __init__(self,dataset,tokenizer,type=4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.type = type
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.type == 4:
            querys_ids_list = [self.tokenizer.encode(text=q,add_special_tokens=True) for q in self.dataset[index].querys]
            return self.dataset[index].fact,querys_ids_list,self.dataset[index].answers
        else:
            fact = self.dataset[index].fact
            crime_set = self.dataset[index].crime_set
            fact_ids = self.tokenizer.encode(text=fact, add_special_tokens=False)
            querys_fact = []
            for query in self.dataset[index].querys:
                query_ids = self.tokenizer.encode(text=query, add_special_tokens=True)
                query_fact = query_ids + fact_ids
                querys_fact.append(query_fact)
            return querys_fact, self.dataset[index].answers, fact_ids, crime_set

class single_style_train_Dataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return  self.dataset[index]

    def __len__(self):
        return len(self.dataset)
def labels2OneHot(labels,class_num=72):
    one_hot_label = []
    for item in labels:
        one_hot = np.zeros(class_num)
        for label_item in item:
            one_hot[label_item] = 1
        one_hot_label.append(one_hot)
    return np.array(one_hot_label)


def collate_fn_v3(batch):
    querys_fact, answers, fact_ids, labels = zip(*batch)
    labels = torch.tensor(labels2OneHot(labels),dtype=torch.float32)
    fact_ids = list(fact_ids)
    attention_mask = []
    seg = []
    max_lengh = 0
    for fact_item in fact_ids:
        if max_lengh<len(fact_item):
            max_lengh = len(fact_item)

    for i,fact_item in enumerate(fact_ids):
        pad = (max_lengh-len(fact_item))*[0]
        fact_ids[i] = [101]+fact_item+pad
        attention_mask.append([2]+len(fact_item)*[1]+[0]*len(pad))
        seg.append([0]*len(fact_ids[i]))

    return torch.tensor(fact_ids),torch.tensor(attention_mask),torch.tensor(seg),labels

def collate_fn_v4(batch):
    fact,querys,answers = zip(*batch)
    # querys_input = tokenizer(querys[0],padding=True, truncation=True, max_length=30, return_tensors='pt')
    fact_input = tokenizer([f for f in fact],padding=True, truncation=True, max_length=2048, return_tensors='pt')
    #fact_input['attention_mask'][:, 0] = 2
    querys = list(querys)
    answers = list(answers)
    max_lengh_q = 0
    max_q_num = 0
    attention_mask = []

    for q in querys:
        if len(q) > max_q_num:
            max_q_num = len(q)
        for q_item in q:
            if len(q_item)>max_lengh_q:
                max_lengh_q = len(q_item)

    # attention_mask seg
    for i, query in enumerate(querys):
        temp_mask = []
        for j, q in enumerate(query):
            temp_mask.append([1] * len(q) + [0] * (max_lengh_q - len(q)))
            pad_q = [0] * (max_lengh_q - len(q))
            query[j] += pad_q
        querys[i] = query
        attention_mask.append(temp_mask)
        if len(query)<max_q_num:
            for j in range(0, max_q_num - len(query)):
                querys[i].append([0] * max_lengh_q)
                attention_mask[i].append([0] * max_lengh_q)

    querys_ids = torch.tensor(querys)
    q_attention_mask = torch.tensor(attention_mask)

    return fact_input,querys_ids,q_attention_mask,answers


def collate_fn_v2(batch):
    querys_fact, answers, fact_ids,_ = zip(*batch)
    querys_fact = list(querys_fact)
    answers = list(answers)
    input_ids = []
    labels = []
    attention_mask = []
    seg = []
    max_lengh_querys_fact = 0
    for i in range(len(querys_fact)):

        labels += answers[i]
        for q in querys_fact[i]:
            if len(q)>max_lengh_querys_fact:
                max_lengh_querys_fact=len(q)


    for i,(querys,fact_id) in enumerate(zip(querys_fact,fact_ids)):
        temp_mask = []
        temp_seg = []
        for j,q in enumerate(querys):
            temp_mask.append([2]*(len(q)-len(fact_id))+[1]*len(fact_id)+[0]*(max_lengh_querys_fact-len(q)))
            temp_seg.append([0]*(len(q)-len(fact_id))+[1]*len(fact_id)+[1]*(max_lengh_querys_fact-len(q)))
            pad_q = [0]*(max_lengh_querys_fact-len(q))
            querys[j]+=pad_q
        input_ids.extend(querys)
        attention_mask.extend(temp_mask)
        seg.extend(temp_seg)

    return torch.tensor(input_ids),torch.tensor(attention_mask),torch.tensor(seg), labels


def collate_fn(batch):
    querys_fact,answers,fact_ids,_ = zip(*batch)
    querys_fact = list(querys_fact)
    answers = list(answers)
    max_lengh_querys_fact = 0
    max_q_num = 0
    max_crime_num = 0
    attention_mask = []
    seg = []
    for q,a in zip(querys_fact,answers):
        if len(q)>max_q_num:
            max_q_num = len(q)
        for a_item in a:
            if len(a_item) > max_crime_num:
                max_crime_num = len(a_item)
        for i,q_item in enumerate(q):
            if len(q_item)>max_lengh_querys_fact:
                if len(q_item)>3000:
                    q[i] = q_item[:3000]
                    max_lengh_querys_fact = 3000
                else:
                    max_lengh_querys_fact = len(q_item)
    #attention_mask seg
    for i,(querys,fact_id) in enumerate(zip(querys_fact,fact_ids)):
        temp_mask = []
        temp_seg = []
        for j,q in enumerate(querys):
            temp_mask.append([2]*(len(q)-len(fact_id))+[1]*len(fact_id)+[0]*(max_lengh_querys_fact-len(q)))
            temp_seg.append([0]*(len(q)-len(fact_id))+[1]*len(fact_id)+[1]*(max_lengh_querys_fact-len(q)))
            pad_q = [0]*(max_lengh_querys_fact-len(q))
            querys[j]+=pad_q
        querys_fact[i] = querys
        attention_mask.append(temp_mask)
        seg.append(temp_seg)

    #pad querys
    for i,Q in enumerate(querys_fact):
                if len(Q) < max_q_num:
                    for j in range(0,max_q_num-len(Q)):
                        querys_fact[i].append([0]*max_lengh_querys_fact)
                        attention_mask[i].append([0]*max_lengh_querys_fact)
                        seg[i].append([0]*max_lengh_querys_fact)

    querys_fact = torch.tensor(querys_fact)
    attention_mask = torch.tensor(attention_mask)
    seg = torch.tensor(seg)


    return querys_fact,attention_mask,seg,answers

    #seg

def load_data(data_dir,batch_size=4,tokenizer=None,type=1):
    if isinstance(data_dir,list):
        train_data = []
        for dir_item in data_dir:
            train_data.extend(torch.load(dir_item+'/new_train_q.pt'))
        val_data = torch.load(data_dir[1] + '/new_dev_q.pt')
    #train_data = torch.load(data_dir+'/train_q.pt')
    else:
        train_data = torch.load(data_dir+'/new_train_q.pt')
        val_data =  torch.load(data_dir + '/new_dev_q.pt')
        test_data = torch.load(data_dir+'/new_test_q.pt')
    train_set = Mutil_def_Dataset(train_data, tokenizer,type = type)
    val_set = Mutil_def_Dataset(val_data, tokenizer,type = type)
    test_set = Mutil_def_Dataset(test_data,tokenizer,type = type)

    if type == 2:
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=False,
                                  collate_fn=collate_fn_v2)
        val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn_v2)
        test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn_v2)
    elif type == 3:
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=False,
                                  collate_fn=collate_fn_v3)
        val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn_v3)

        test_loader = DataLoader(test_set,batch_size=1,collate_fn=collate_fn_v3)
    elif type == 4:
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=False,
                                  collate_fn=collate_fn_v4)
        val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn_v4)
        test_loader = DataLoader(test_set,batch_size=batch_size,collate_fn=collate_fn_v4)
    else :
        train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size, drop_last=False,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn)

    return train_loader,val_loader,test_loader

if __name__ == "__main__":


    data_dir = './mutil_defendant_mutil_crime/new_processed'
    save_data = "mutil_defendant_mutil_crime/new_processed/single_style_train"

    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir, batch_size=1,tokenizer=tokenizer, type=1)
    train_data_list = []
    for item in train_dataloader:
        token_len = item[0].size()[-1]
        input_ids = item[0].view(-1, token_len)
        attention_mask = item[1].view(-1, token_len)
        seg = item[2].view(-1, token_len)
        labels = labels2one_hot(item[3], num_classes=72)
        labels = labels.view(-1,72)
        for i in range(input_ids.size()[0]):
            data_item = {}
            data_item["input_ids"] = input_ids[i]
            data_item["attention_mask"] = attention_mask[i]
            data_item["token_type_ids"] = seg[i]
            data_item["label"] = labels[i]
            train_data_list.append(data_item)
    for i, data_dict in enumerate(train_data_list):
        torch.save(data_dict, f'mutil_defendant_mutil_crime/new_processed/single_style_train/data_{i}.pt')


