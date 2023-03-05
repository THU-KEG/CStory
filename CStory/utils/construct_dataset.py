from hashlib import new
from typing import List
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from sklearn.cluster import KMeans, Birch,SpectralClustering,AgglomerativeClustering
from collections import defaultdict
import time
import codecs
import re
import hdbscan
import jieba
import datetime
import jieba.posseg as pseg
jieba.enable_paddle()
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import normalized_mutual_info_score,homogeneity_score,completeness_score,adjusted_rand_score,fowlkes_mallows_score,adjusted_mutual_info_score,silhouette_score
from news_vector import get_vectors_interface
from numpy import linalg as LA
from RDBSCAN import RDBSCAN
from DBSCAN import DBSCAN
from INDBSCAN import INDBSCAN
from INDBSCAN import NDBSCAN
import random
from SINGLEPASS import SINGLEPASS
import argparse
import pandas as pd
from sklearn import metrics
from sentence_transformers import SentenceTransformer,util
import jieba
import jieba.analyse

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(root_dir,'model_bm25'))
from CalculateScore import BM25

stopwords = ['直播吧','中新社','新华社','人民网','人民日报','人民日报社','中新','来源','客户端','中新经纬',
                '新华网','网易体育',"观察者网","朋友圈直播吧","扫码","微信","微信公众号","华龙网","中新网","中国侨网","东南网","搜狐网"
                "央视网","东方网","大河网讯","大河网","讯篮网",'新浪财经','文汇通网','央广网','央视网','观察者网',"朋友圈"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='your input file', default='/data/skj/information_flow/data_unprocessed/real_world_data/raw_2021-05-20_2021-06-02.json')
    parser.add_argument("-e", "--encode", default='sentence_transformer')
    parser.add_argument("-c", "--cluster", default='SINGLEPASS')
    parser.add_argument("-o", "--output_dir", help = "your output directory")
    parser.add_argument('--period', default=20)
    parser.add_argument('--output_path', default='tmp.tsv')
    parser.add_argument("-n", "--num", help = "pre-defined cluster numbers", default=2)
    parser.add_argument("-t", "--threshold", help = "single pass threshold to split a cluster", default=0.7)
    parser.add_argument("-p", "--point", help = "min num points in a  cluster for dbscan", default=2)
    parser.add_argument("-r", "--radius", help = "radius of cluster for dbscan", default=0.7)
    parser.add_argument("--entity_number", help="entity number to identify two news", default=2)
    parser.add_argument("--ratio", help='Positive-negative sample ratio', default=2)
    parser.add_argument("--type", default='story')
    parser.add_argument("--max_samples_per_story", default=100)
    args = parser.parse_args()
    if args.radius:
        args.radius = float(args.radius)
    if args.point:
        args.point = int(args.point)
    if args.num:
        args.num = int(args.num)
    if args.threshold:
        args.threshold = float(args.threshold)
    if args.cluster and args.cluster not in ["DBSCAN","RDBSCAN","HDBSCAN", "KMEANS", "SINGLEPASS",
    "TREEDBSCAN","EDBSCAN", "NDBSCAN", "INDBSCAN","BIRCH", "DENSTREAM",'SPECTRAL','AGG']:
        print("聚类算法错误")
        return 
    if args.encode and args.encode not in ['sentence_transformer', 'word2vec', 'bertvec','skip_gram']:
        print("编码方法错误")
        return 

    return args


def read_news_list(data_file):
    with open(data_file, 'r') as file:
        news_list = json.load(file)
    titles = set()
    res = []
    for news in news_list:
        if news['title'] not in titles:
            titles.add(news['title'])
            res.append(news)
    #print("num of news:", len(res))
    return res


def cluster_process(args, news_vec, news_list=None):
    labels = []
    #sent_model = 
    cluster_method = args.cluster
    if cluster_method == 'DBSCAN':
        labels = DBSCAN(args.radius, news_vec, args.point)

    elif cluster_method == "BIRCH":
        labels = Birch(n_clusters=args.num).fit_predict(news_vec)

    elif cluster_method == "RDBSCAN":
        labels = RDBSCAN(args.radius, news_vec, args.point, news_list)

    elif cluster_method == "AGG":
        labels = AgglomerativeClustering(n_clusters=2).fit_predict(news_vec)

    elif cluster_method == 'SPECTRAL':
        labels = SpectralClustering( n_clusters=args.num,random_state=0).fit_predict(news_vec)

    elif cluster_method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.point)
        #news_vec = news_vec.reshape(-1,1)
        labels = clusterer.fit_predict(news_vec)
        max_index = np.max(labels)
        max_index += 1
        for order,label in enumerate(labels):
            if label == -1:
                labels[order] = max_index
                max_index += 1

    elif cluster_method == 'INDBSCAN':
        labels = INDBSCAN(news_vec, args.radius, args.point)
        
    elif cluster_method == 'KMEANS':
        labels = KMeans(n_clusters=args.num, random_state=9).fit_predict(news_vec)

    elif cluster_method == 'SINGLEPASS':
        label_cluster = SINGLEPASS(news_vec, threshold = args.threshold)
        convert_labels = len(news_vec) * [-1]
        for key in label_cluster.keys():
            for value in label_cluster[key]:
                convert_labels[value] = key
        #print("SP",convert_labels)
        labels = convert_labels

    return labels



def print_binary_dataset(args, dataset_list, non_redundant_stories):
    
    output_file = open(args.output_path, 'w')
    story_file = open('tmp_story.txt','w')
    for story in non_redundant_stories:
        for news in story:
            story_file.write(news['title']+'\n')
        story_file.write('\n\n')

    output_file.write("title1\ttitle2\tlabel\n")
    for data in dataset_list:
        #print(data)
        title1, title2, label = data[0], data[1], '1'
        a = title1[:256].replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        b = title2[:256].replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        output_file.write("{}\t{}\t{}\n".format( a, b, label))

def isMatch(list_a, list_b):
    tmp = set(list_a) & set(list_b)

    return len(tmp) > 0

def filter_redundant_stories(args, stories):
    res = []
    args.threshold = 0.8
    for story in stories:
        new_story = []
        if len(story) <= 1:
            continue
        #得到向量
        #print('story', len(story))
        story_vec = get_vectors_interface(method=args.encode,news_list = story, seq_len=512)
        #开始聚类
        story_pred = cluster_process(args, story_vec)
        story_dict = defaultdict(list)
        for index, label in enumerate(story_pred):
            story_dict[label].append(story[index])

        for key,value in story_dict.items():
            new_story.append(value[0])
        
        res.append(new_story)
    return res


def cluster_interface(args,news_list):
    news_vec = get_vectors_interface(method=args.encode,news_list = news_list, seq_len=512)
    y_pred = cluster_process(args, news_vec)
    #print('y_pred',y_pred)
    return y_pred

def story_cluster(args):
    #读取数据
    news_list = read_news_list(args.input_file)
    #得到向量
    news_vec = get_vectors_interface(method=args.encode,news_list = news_list, seq_len=512)
    #开始聚类
    args.threshold = 0.6
    y_pred = cluster_process(args, news_vec)
    clusters = defaultdict(list)
    for index, label in enumerate(y_pred):
        clusters[label].append(news_list[index])

    cluster_array = [list(i) for i in clusters.values()]
    return cluster_array


def get_first_sentence(s):
    res = ''
    count = 3
    for char in s:
        if char != '。':
            res += char
        else:
            if count <= 1:
                break
            else:
                count -= 1  
    return res


w_f = open('entity.txt','w')
def get_key_entities(story):
    #sentences = [i['title']+i['content'][:100] for i in story]

    sentences = [i['title']+ get_first_sentence(i['content']) for i in story]
    sentences = [sentence.replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','').replace("北京时间",'') for sentence in sentences]
    res = []
    for sentence in sentences:
        words = pseg.cut(sentence,use_paddle=True)
        
        tmp = []
        for word, flag in words:
            if flag in ['LOC','PER','ORG','nz','nw'] and word not in stopwords:
                tmp.append(word)

        tmp = list(set(tmp))
        res.append(tmp)
        print(tmp)
        w_f.write(' '.join(tmp)+'\n')
    return res

def make_event_thread_pairs(args, stories):
    #BM25_model = BM25()
    res = []
    period = args.period
    #print(stories)
    
    for story in stories:
        story = sorted(story, key=lambda x:x['publishTime'])
        key_sentences = get_key_entities(story)
        #BM25_model.read_news_list(story)
        #BM25_model.calculate_BM25()
        #results, lists, key_sentences = BM25_model.get_result()
        #print(results)
        #assert lists == story
        
        for index, news in enumerate(story):
            begin_pos = index - period
            if begin_pos < 0:
                begin_pos = 0
            #begin_pos = 0
            #candidate_score = results[index][begin_pos:index]
            for candidate_index in range(index-begin_pos):   
                # if candidate_score[candidate_index] > 40 or candidate_score[candidate_index] < 10:
                #     continue 
                
                if not isMatch(key_sentences[index],key_sentences[begin_pos+candidate_index]):
                    continue  

                else:
                    #title1 = lists[begin_pos+candidate_index]['title'] + lists[begin_pos+candidate_index]['content'][:256]
                    #title2 = lists[index]['title'] + lists[index]['content'][:256]
                    title1 = story[begin_pos+candidate_index]['title']+story[begin_pos+candidate_index]['content'][:256]
                    title2 = story[index]['title']+story[index]['content'][:256]
                    res.append([title1, title2])
    #print('res', res)
    return res

if __name__ == '__main__':
    args = parse_args()
    stories = story_cluster(args)
    non_redundant_stories = filter_redundant_stories(args, stories)
    datasets = make_event_thread_pairs(args, non_redundant_stories)
    print_binary_dataset(args, datasets, non_redundant_stories)
    #normal_cluster(args)