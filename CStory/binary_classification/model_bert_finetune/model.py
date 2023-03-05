import json
from collections import defaultdict
import torch
import torch.nn as nn
import os
from sklearn.metrics import f1_score

import sklearn
import torch.nn.functional as F
import re
from transformers import BertTokenizer
from transformers import BertModel
#from transformers import BertForMaskedLM

def creat_model(args):
    model = BertModel.from_pretrained(args.model_path)     
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    return model, tokenizer

class FinetuneModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.threshold = 0.5
        self.device = args.device
        self.model, self.tokenizer = creat_model(args)
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.embedding = self.model.get_input_embeddings()
        self.template_type = args.template_type
        self.hidden_size = self.embedding.embedding_dim
        self.fc = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, 2))


        self.recall = 0
    
    def get_query(self, data):
        def add_tokens(sent1):  
            sent1 = '[CLS]' + sent1 + '[SEP]'
            sent1 = self.tokenizer.tokenize(sent1)[:max_len]
            sent1 = sent1 + ['[PAD]'] * (max_len - len(sent1))
            sent1 = self.tokenizer.convert_tokens_to_ids(sent1)
            return sent1


        #if self.task_type == 'normal':
        title1 = data['title1']
        title2 = data['title2']
        max_len = self.max_length
        res = []
        if self.template_type == 'base':
            for index in range(len(title1)):
                #print('title1',title1[index])
                title1[index] = title1[index][:(max_len//2-2)]
                title2[index] = title2[index][:(max_len//2-2)]
                #print('title2', title2[index])
                sent1 = title1[index] + '[SEP]' + title2[index] 
                res.append(add_tokens(sent1))

        elif self.template_type == 'same_event':
            for index in range(len(title1)):
                sent1 = title1[index] + '和' + title2[index] + '是同一事件。'
                res.append(add_tokens(sent1))

        elif self.template_type == 'story_line':
            for index in range(len(title1)):
                sent1 = '在' + title1[index] + '之后发生了' + title2[index] + '事件。'
                res.append(add_tokens(sent1))
        #print(len(res))                
        #print(res)
        #for i in res:
        #    print(len(i))
        return torch.LongTensor(res)            

    def forward(self, data):
        batch_size = len(data['label'])
        queries = self.get_query(data)
        labels = data['label']
        labels = torch.LongTensor([int(i) for i in labels])
        output = self.model(queries.to(self.device))
        
        output = self.fc(output.pooler_output)
        #print(output)
        labels_one_hot = torch.zeros(batch_size, 2).scatter_(1, labels.unsqueeze(1), 1).to(self.device)
        
        loss =  -1 * torch.sum(labels_one_hot * F.log_softmax(output, dim=1))
        
        #pred_labels = (F.softmax(output,dim=1)[:,-1] > self.threshold).long()
        pred_labels = (output[:,-1]>self.threshold).long()
        
        labels = labels.to(self.device)
        #print("HHH", f1_score(labels, pred_labels))
        hit1 = int(torch.sum(pred_labels == labels))
        
        pred_positive = 0 
        true_positive_in_pred = 0

        label_positive = 0
        recall_positive_in_label = 0

        for i in range(len(pred_labels)):
            if pred_labels[i] == 1:
                pred_positive += 1
                if labels[i] == 1:
                    true_positive_in_pred += 1

            if labels[i] == 1:
                label_positive += 1
                if pred_labels[i] == 1:
                    recall_positive_in_label += 1



        for i in range(len(labels)):
            if pred_labels[i] == 1:
                self.recall += 1
        #self.recall += int(torch.sum((pred_labels==1) == (labels==1)))
        return loss , hit1, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive

    def get_logits(self, data):
        queries = self.get_query(data)
        output = self.model(queries.to(self.device))
        output = self.fc(output.pooler_output)
        output = F.softmax(output, dim=1)
        return output 

