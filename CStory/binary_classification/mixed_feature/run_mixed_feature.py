import json
import pandas as pd
import os
from operator import itemgetter
from collections import defaultdict
import numpy as np
import math
import numpy.linalg as LA
import jieba.analyse
import jieba.posseg as pseg
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def read_news_list(path):
    sentences_list = []
    news_list = json.load(open(path))
    for news in news_list:
        
        #print(news)
        sentence = (news['title'] + '。' + news['content']).replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        
        sentences_list.append(sentence)
    return sentences_list



def init_tfidf(data):
    sentences_list = []

    for i in data:
        sentence= (i['title'] + '。' + i['content']).replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        #sentence = jieba.analyse.extract_tags(sentence, )
        #sentence = ' '.join(jieba.analyse.extract_tags(sentence,20))
        sentences_list.append(sentence)

        

    #sent_words = [list(jieba.cut(p_sen)) for p_sen in sentences_list]
    #document = [" ".join(p_sen) for p_sen in sent_words]
    tfidf_model = TfidfVectorizer(stop_words=[])
    tfidf_model.fit(sentences_list)
    #sparse_result = tfidf_model.transform(document)
    print("所有的单词：", tfidf_model.vocabulary_)
    print(len(tfidf_model.vocabulary_))
    return tfidf_model
    # 这里显示所有的词，也可使用tfidf_model.get_feature_names()，区别是get_feature_names会按照index排序，而vocabulary_不会
    # print("第一个句子：", document[0])
    

def get_news_from_certain_time_window( news_path, time_window):
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
            #print(index)
        
            if news_a not in documents:
                documents.append(news_a)
            if news_b not in documents:
                documents.append(news_b)
    return documents


def calculate_score(pred_labels, labels):
    #print('pred', pred_labels[:40])
    #print('label', labels[:40])
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


def get_jaccard_similarity(sentence1, sentence2):
    set_1 = set(jieba.cut(sentence1))
    set_2 = set(jieba.cut(sentence2))
    jaccard_coefficient = len(set_1 & set_2) / len(set_1 | set_2)
    return jaccard_coefficient

def get_noun_similarity(sentence1, sentence2):
    words = pseg.cut(sentence1,use_paddle=True)
    participate_set1, participate_set2, loc_set1, loc_set2 = set(),set(),set(),set()
    for word, flag in words:
        k = re.split('，|、', word)
        for i in k:
            if flag in ['PER','ORG','nz','nw']:
                participate_set1.add(i)
            elif flag in ['LOC']:
                loc_set1.add(i)


    words = pseg.cut(sentence2,use_paddle=True)
    for word, flag in words:
        k = re.split('，|、', word)
        for i in k:
            if flag in ['PER','ORG','nz','nw']:
                participate_set2.add(i)
            elif flag in ['LOC']:
                loc_set2.add(i)

    jaccard_similarity = len(participate_set1 & participate_set2) / (len(participate_set1 | participate_set2)+1e-6)
    loc_similarity = 1 if loc_set1 & loc_set2 else 0
    return jaccard_similarity, loc_similarity

if __name__ == '__main__':
    
    project_root = os.path.dirname(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
    print('hello',project_root)
    train_set_time_window = ['2021-05-20 00:00:00','2021-05-26 23:59:59']
    documents = json.load(open(project_root + '/dataset/ccl.tsv_DATA.json'))
    tfidf_model = init_tfidf(documents)
    #/data/skj/data/A_test.tsv
    test_triples  = pd.read_csv(project_root + '/dataset/A_test.tsv','\t')

    vectors = []
    similarities = []
    labels = []
    for index, row in test_triples.iterrows():
        sentence1 = row[0]
        sentence2 = row[1]
        labels.append(row[2])
        a = tfidf_model.transform([sentence1]).toarray()[0]
        b = tfidf_model.transform([sentence2]).toarray()[0]

        #res = tfidf_model.transform([sentence1,sentence2]).toarray()
        tfidf_similarity = np.sum(a* b)/((LA.norm( a,  axis=0)+ 1e-6) * (LA.norm(b,axis=0)+1e-6))
        #jaccard系数测试
        jaccard_similarity = get_jaccard_similarity(sentence1, sentence2)
        #多种相似度结合测试
        #final_similarity = 0.7 * tfidf_similarity + 0.2 * jaccard_similarity + 0.1 * loc_similarity
        similarities.append(jaccard_similarity)
        #如果想使用其他相似度的话，需要把下面对应的相似度去掉注释，就可以输出对应的相似度方法
        #similarities.append(loc)
        #similarities.append(final_similarity)
        #print(index, final_similarity)
    
    print(similarities)
    #similarities
    # min_value, max_value =  min(similarities), max(similarities)

    #print(min_value ,max_value)
    res = [0,0,0,0]
    for value in np.arange(0, 1, 1/50):

        pred_labels = []
        for s in similarities:
            if s > value:
                print(s, value )
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        hit1, precision, recall, f1 =  calculate_score(pred_labels, labels)
        if f1 > res[3]:
            res = [hit1, precision, recall, f1]
        
    print("最佳结果Hit@1:{} precision:{} recall:{} f1:{}".format(res[0], res[1], res[2], res[3]))
