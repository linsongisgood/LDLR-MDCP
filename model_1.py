import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer,AutoConfig
from multi_q_transformer import MQLayer
class Charge_Fusion(nn.Module):
    def __init__(self,config):
        #Q_fact shape [batch,hidden_state] charge shape [cls_num,sequence_len,hidden_state]
        super(Charge_Fusion, self).__init__()
        self.fact_linear = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.charge_def_linear = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.fusion_linear = nn.Linear(config['hidden_size'],config["hidden_size"])
        self.Ws = nn.Parameter(torch.Tensor(config['num_labels'], config['hidden_size']).uniform_(-0.25, 0.25),requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(config['hidden_size']).uniform_(-0.25, 0.25),requires_grad=True)

    def forward(self,Q_fact,charge,charge_mask):
        cls_num,seq,hid = charge.size()
        q_num,hid = Q_fact.size()
        Q_fact = self.fact_linear(Q_fact)
        charge = self.charge_def_linear(charge)
        #charge shape [cls_num,hid,seq]
        attention_sore = torch.matmul(Q_fact.unsqueeze(dim=0),charge.permute(0,2,1)).permute(1,0,2)
        masked_attention_score = attention_sore + (1 - charge_mask) * (-1e9)
        masked_attention_score = nn.Softmax(dim=-1)(masked_attention_score)
        charge_emb = torch.matmul(masked_attention_score.unsqueeze(dim = 2),charge).squeeze()
        fusion_charge_fact = Q_fact.unsqueeze(dim = 1)+charge_emb
        fusion_charge_fact = torch.tanh(self.fusion_linear(fusion_charge_fact))
        fusion_charge_fact = fusion_charge_fact*self.Ws+self.bias

        return torch.sum(fusion_charge_fact,dim=-1)


class MutilDefendantCrimePre(nn.Module):
    def __init__(self,pretrain_path = None,config=None,cus_config = None):
        super(MutilDefendantCrimePre, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrain_path)
        self.cus_config = cus_config
        self.encoder = AutoModel.from_pretrained(pretrain_path,config=self.config)
        self.ATrans_decoder = nn.ModuleList([MQLayer(config, self.cus_config) for _ in range(self.cus_config["n_qlayer"])])
        self.classifier = ClassificationHead(cus_config)
        self.dropout = nn.Dropout(0.2)
        if self.cus_config['charge_fusion']:
            self.charge_fusion_layer = Charge_Fusion(config = self.cus_config)

    def forward(self,fact_input,querys_ids,q_mask,charge_def_emb = None,charge_mask = None):
        batch,q_num,token_len = querys_ids.size()[0],querys_ids.size()[1],querys_ids.size()[2]
        fact_output = self.encoder(**fact_input)[0]
        querys_output = self.encoder(input_ids=querys_ids.view(-1,token_len),attention_mask=q_mask.view(-1,token_len))[0]
        querys_output = querys_output.view(batch,q_num,token_len,self.cus_config['hidden_size'])
        querys_emb = querys_output
        extend_attention_mask = self.get_attention_mask(fact_input['attention_mask'])
        fact_hidden_state = self.dropout(fact_output)

        batch, q_num = querys_emb.size()[0], querys_emb.size()[1]
        fact_len, hidden_size = fact_hidden_state.size()[1], fact_hidden_state.size()[-1]
        fact_hidden_states = fact_hidden_state.unsqueeze(dim=1).repeat(1, q_num, 1, 1).view(-1, fact_len, hidden_size)
        attention_mask = extend_attention_mask.repeat(1, q_num, 1, 1).view(-1, 1, fact_len).unsqueeze(dim=2)
        #attention_mask = extend_attention_mask
        question = querys_emb.view(-1, token_len,hidden_size)

        for i, mqlayer in enumerate(self.ATrans_decoder):
            hidden_state = mqlayer(question, fact_hidden_states,attention_mask)
        hidden_state = self.dropout(hidden_state)

        if charge_def_emb is not None and charge_mask is not None:
            result = self.charge_fusion_layer(hidden_state.mean(1),charge_def_emb,charge_mask)
        else :
            result = self.classifier(hidden_state.mean(1))

        return result

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids or attention_mask"
                )
        try:
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        except:
            print(extended_attention_mask)
            exit()
        # extended_attention_mask = ~extended_attention_mask * -10000.0
        return extended_attention_mask
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(config["hidden_size"], config["num_labels"])
    def forward(self, hidden_states):
        if hidden_states.dim()==3:
            hidden_states = hidden_states[:, 0]  # take <s> token (equiv. to [CLS# ])
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)

        return output

