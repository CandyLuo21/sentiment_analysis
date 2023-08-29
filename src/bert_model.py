import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class BertModel(nn.Module):
    def __init__(self, num_labels, model_name, dropout):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        #add dropout for regularization
        self.bert_drop = nn.Dropout(dropout)
        # #only one output , 768 is hidden layout for bert-base
        self.out = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        #Bert default settings return 2 outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print("BertModel forward outputs:", outputs)
        #dropout
        # bo = self.bert_drop(outputs)
        logits = outputs.logits
        # print("BertModel forward logits:", logits)
        return logits
