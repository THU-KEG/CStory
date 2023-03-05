#coding=utf-8
__author__ = 'root'
from email.policy import default
from torch import threshold

from transformers import DetrFeatureExtractor
#from PMI import *
import os
import jieba
import json
from random import randint
import jieba.analyse
import jieba.posseg as pseg
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import math

class MI:
    def __init__(self, documents) -> None:#documents是一个文本列表
        self.news_list = documents
        self._length = len(documents)

        self.id2news = self._build_news_index(documents)
        
        self.word2docid, self.docid2word = self._init_dict(documents)


        
    
    def _build_news_index(self, documents):
        id2news = defaultdict()
        for news in documents:
            id2news[news['newsID']] = news 
        return id2news
        

    def _cut_sentence(self, text):
        #res = jieba.lcut(text)
        res =  jieba.analyse.extract_tags(text,30)
        return  res

    def _init_dict(self, documents):
        docid2word = defaultdict(set)
        word2docid = defaultdict(set)
        for index, doc in enumerate(documents):
            text = doc['title'] + '。' + doc['content'].replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')

            words = self._cut_sentence(text)
            doc_id = doc['newsID']

            
            for word in words:
                docid2word[doc_id].add(word)
                word2docid[word].add(doc_id)
            
        return  word2docid, docid2word

    def calculate_pmi(self, joinpercent, wordpercent1, wordpercent2):
        return joinpercent * math.log(joinpercent/(wordpercent1*wordpercent2+ 1e-6)+ 1e-6)

    def calculate_similarity(self, doc1, doc2):
        text1 = doc1.replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        text2 = doc2.replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        wordbag_1 = self._cut_sentence(text1)
        wordbag_2 = self._cut_sentence(text2)

        res = 0
        for a in wordbag_1:
            for b in wordbag_2:
                if a == b:
                   continue
                
                joinpercent = len(self.word2docid[a] & self.word2docid[b])
                a_cocurrence = len(self.word2docid[a])
                b_cocurrence = len(self.word2docid[b])
                pmi_a_b = self.calculate_pmi(joinpercent, a_cocurrence, b_cocurrence)
                res += pmi_a_b
        return res


    def calculate_score(self,pred_labels, labels):
        print('pred', pred_labels[:40])
        print('label', labels[:40])
        hit1 = np.sum(np.array(pred_labels) == np.array(labels))
        
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
        
        precision = true_positive_in_pred / (pred_positive + 1e-6) 
        recall = recall_positive_in_label / (label_positive + + 1e-6)
        f1 = 2 * precision * recall /(precision + recall + 1e-6)				
        hit1 /= len(labels)
        print("Hit@1:{} precision:{} recall:{} f1:{}".format(hit1, precision, recall, f1))
        return hit1, precision, recall, f1

    def random_score(self, tsv_path):
        df = pd.read_csv(tsv_path, '\t')

        labels = []
        for index,row in df.iterrows():
            labels.append(row[-1])


        count = 50
        total_hit1, total_precision, total_recall, total_f1 = 0 ,0 ,0 ,0
        for _ in range(count):
            pred_labels = []
            for _ in range(len(df)):
                pred_labels.append(randint(0,1))
            hit1, precision, recall, f1 =   self.calculate_score(pred_labels, labels)


            total_hit1 += hit1
            total_precision += precision
            total_recall += recall
            total_f1 += f1

        res= [total_hit1/count, total_precision/count, total_recall/count, total_f1/count]
        print("最佳结果：Hit@1:{} precision:{} recall:{} f1:{}".format(res[0], res[1], res[2], res[3]))


    def read_and_process_tsv(self, tsv_path):
        df =  pd.read_csv(tsv_path,'\t')
        labels = []
        pred_labels = []
        similarities = []
        for index ,row in df.iterrows():
            labels.append(row[2])
            #print('index', index)
            doc1 = row[0]
            doc2 = row[1]
            similarity = self.calculate_similarity(doc1, doc2)
            similarities.append(similarity)
            print(doc1[:50]+ ' ' + doc2[:50] + ' ' + str(similarity))

        
        #similarities = similarities
        min_value, max_value =  int(min(similarities)), int(max(similarities))

        res = [0,0,0,0]
        for value in range(min_value, max_value, int((max_value-min_value)/50) ):
            pred_labels = []
            for s in similarities:
                if s > value:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)
            hit1, precision, recall, f1 =   self.calculate_score(pred_labels, labels)
            if f1 > res[3]:
                res = [hit1, precision, recall, f1]
        
        print(" Hit@1:{} precision:{} recall:{} f1:{}".format(res[0], res[1], res[2], res[3]))
        






def get_news_from_certain_time_window(tsv_path, news_path, time_window):
    documents = []
    news_list = json.load(open(news_path))
    id2text = {}

    for news in news_list:
    #tmp = (tmp['title']+ '。' + tmp['content']).replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        id2text[news['newsID']] = news 

    def time_available(news, time_window):
        return time_window[0] <= news['publishTime'] and news['publishTime'] <= time_window[1]


    project_root = os.path.dirname(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
    relation_triples = pd.read_csv(project_root + '/dataset/ccl.tsv_DATA.csv')

    for index, row in relation_triples.iterrows():
        id_a = row[0]
        news_a = id2text[id_a]       
        id_b = row[1]
        news_b = id2text[id_b]  
        if  time_available(news_a, time_window) and time_available(news_b, time_window): 
            print(index)
        
            if news_a not in documents:
                documents.append(news_a)
            if news_b not in documents:
                documents.append(news_b)
    return documents



        

if __name__ == '__main__':
     
    train_set_time_window = ['2021-05-20 00:00:00','2021-05-26 23:59:59']
    #documents = get_news_from_certain_time_window('/data/skj/data/ccl.tsv_DATA.csv','/data/skj/data/ccl.tsv_DATA.json',train_set_time_window)
    project_root = os.path.dirname(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
    documents = json.load(open(project_root + '/dataset/ccl.tsv_DATA.json'))
    a = MI(documents)
    #/data/skj/data/A_test.tsv
    # a.read_and_process_tsv('/data/skj/data/100news_test.tsv')
    a.random_score(project_root + '/dataset/A_test.tsv')


    
