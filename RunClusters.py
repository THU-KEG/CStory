from hashlib import new
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms'))
from sklearn.cluster import KMeans, Birch,SpectralClustering,AgglomerativeClustering
from collections import defaultdict
import time
import codecs
import re
import hdbscan
import datetime
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='your input file')
    parser.add_argument("-i", "--input_dir", help="your input directory")
    parser.add_argument("-e", "--encode", default='sentence_transformer')
    parser.add_argument("-c", "--cluster", default='SINGLEPASS')
    parser.add_argument("-o", "--output_dir", help = "your output directory")
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



def read_stories(data_file):
    with open(data_file, 'r') as file:
        stories_list = json.load(file)

    return stories_list

def print_binary_dataset(args, dataset_list):
    output_file = open(args.output_path, 'w')
    output_file.write("title1\ttitle2\tlabel\n")
    for data in dataset_list:
        title1, title2, label = data[0], data[1], data[2]
        a = (title1['title']+'。'+title1['content'])[:256].replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        b = (title2['title']+'。'+title2['content'])[:256].replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
        output_file.write("{}\t{}\t{}\n".format( a, b, label))



def print_stories(args, labels, news_list):
    cluster = defaultdict(list)

    for i, label in enumerate(labels):
        cluster[label].append(i)

    array = []
    for label, index_array in cluster.items():
        story = []
        for idx in index_array:
            story.append(news_list[idx])

        if len(story) <= 1:
            continue
        array.append(story)

    json.dump(array, open(args.output_path, 'w'))


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def evaluate_result(y_pred, y_true):
    # print("调整兰德系数:",adjusted_rand_score(y_pred,y_true))
    # aaa_file = open('knmi.txt', 'a+')
    # aaa_file.write(',{}'.format(adjusted_mutual_info_score(y_pred,y_true))) 

    # print("NMI",normalized_mutual_info_score(y_pred,y_true))

    h = homogeneity_score(y_pred,y_true)
    print("同质性:",h)
    c = completeness_score(y_pred,y_true)
    print("完整性",c) 
    # print("轮廓系数",silhouette_score(news_vec, labels, metric='cosine'))
    print("V-measure", 2 * h * c /(h + c))
    # print("FMI",fowlkes_mallows_score(y_pred,y_true))
    # p = purity_score(y_true, y_pred)
    # print("purity", p)
    # label_num = set()
    # for l_index,label in enumerate(labels):
    #     label_num.add(label)

def isMatch(news_a, news_b, threshold):
    content_a = news_a['title'] + news_a['content'][:256]
    
    content_b = news_b['title'] + news_b['content'][:256]     
    
   
    keyword_a = jieba.analyse.extract_tags(content_a, topK=10, allowPOS=('ns','nr','nt'))
    keyword_b = jieba.analyse.extract_tags(content_b, topK=10, allowPOS=('ns','nr','nt'))

    count = 0
    for word_a in keyword_a:
        for word_b in keyword_b:
            if word_a == word_b:
                count += 1 
    
    if threshold <= count:
        print('content_a', content_a)
        print('content_b', content_b)
        print('keyword_a', keyword_a)
        print('keyword_b', keyword_b)
    return threshold <= count
    

def select_same_event_news(args,array):
    random.shuffle(array)
    res = []
    for i in range(len(array)):
        for j in range(i+1, len(array)):
            #print('sdfsdf', args.entity_number)
            if isMatch(array[i], array[j], args.entity_number):
                res.append([array[i], array[j], 1])
            if len(res) > args.max_samples_per_story:
                return res
    return res

def select_different_event_news(values, array, num):
    res = []
    for other_array in values:
        if other_array == array:
            continue
        for o  in other_array:
            for a in array:
                res.append([o, a, 0])
                if len(res) > num:
                    return res
    return res

def convert_time_format(times):
    for time in range(1, times):
        index = 0
        while index < len(time) and  time[index] == '0':
            index += 1
        time = time[index:]

def timeDurtion(time_a, time_b):
    return True
    split_time_a = convert_time_format(time_a.split('-| |:', time_a))
    split_time_b = convert_time_format(time_b.split('-| |:', time_b))
    split_time_a =  datetime.datetime(*split_time_a)
    split_time_b = datetime.datetime(*split_time_b)
    if (split_time_b-split_time_a).total_seconds > 43200:#12个小时之外
        return True 
    else:
        return False

def select_same_story_line_news(args, array):
    res = []
    for i in range(len(array) -1):
        #if timeDurtion(array[i]['publishTime'], array[i+1]['publishTime']):
        if isMatch(array[i], array[i+1], args.entity_number):
            res.append([array[i], array[i+1], 1])
    return res 

def select_different_story_line_news(array, stories, num):
    res = []
    while len(res) < num:
        story = random.choice(stories)
        if array == story:
            continue
        other_news = random.choice(story)
        news = random.choice(array)
        res.append([other_news, news, 0])
        #if index > 5 * len(array): 
        #    return res
    return res

def select_another_key(keys, key):
    while True:
        random_key = random.choice(keys)
        if random_key != key:
            break
    return random_key


def add_new_data(args, event_dict, same_event_dataset, story_line_dataset, story_index, story, stories):
    keys = list(event_dict.keys())
    values = list(event_dict.values())

    event_list = []
    for key,value in zip(keys, values):
        if len(value) > 1:
            positive_samples = select_same_event_news(args,value)
            same_event_dataset.extend(positive_samples)
        
            negative_events = select_different_event_news(values, value, args.ratio * len(positive_samples))
            same_event_dataset.extend(negative_events)

        single_news = random.choice(value)
        event_list.append(single_news)

    event_list = sorted(event_list, key=lambda x : x['publishTime'])
    positive_stories = select_same_story_line_news(args, event_list)
    story_line_dataset.extend(positive_stories)

    #another_key = select_another_key(story_index, range(len(stories)))
    negative_samples = select_different_story_line_news(story, stories, 2 * len(positive_stories))
    story_line_dataset.extend(negative_samples)

def normal_cluster(args):
    news_list = json.load(open(args.input_file))
    news_vec = get_vectors_interface(method=args.encode,news_list = news_list, seq_len=512)
    pred = cluster_process(args, news_vec, news_list)
    res_dict =defaultdict(list)
    for index,label in enumerate(pred):
        res_dict[label].append(news_list[index]['title'])
    
    w_f = open('tmp.txt', 'w')
    res_list = sorted(list(res_dict.items()), key=lambda i:len(i[1]), reverse=True)
    for label, titles in res_list:
        print('cluster' + ' '+ str(label), file=w_f)
        for title in titles:
            print(title, file=w_f)
        print(file=w_f)


def event_cluster(args):
    files = os.listdir(args.input_dir)
    all_same_event_dataset = []
    all_story_line_dataset = []
    for f in files:
        same_event_dataset = []
        story_line_dataset = []
        stories = read_stories(os.path.join(args.input_dir, f))
        for story_index, story in enumerate(stories):
            if len(story) <= 1:
                continue
            #得到向量
            print('story', len(story))
            story_vec = get_vectors_interface(method=args.encode,news_list = story, seq_len=512)
            #开始聚类
            story_pred = cluster_process(args, story_vec)
            
            event_dict = defaultdict(list)
            event_list = []
            for index,label in enumerate(story_pred):
                event_dict[label].append(story[index])

            add_new_data(args,event_dict, same_event_dataset, story_line_dataset, story_index, story, stories)

        all_same_event_dataset.extend(same_event_dataset)
        all_story_line_dataset.extend(story_line_dataset)
        args.output_path = os.path.join(args.output_dir, "same_event_" + f)
        print_binary_dataset(args, same_event_dataset)
        args.output_path = os.path.join(args.output_dir, "story_line_" + f)
        print_binary_dataset(args, story_line_dataset)

    args.output_path = os.path.join(args.output_dir, "same_event.txt")
    print_binary_dataset(args, all_same_event_dataset)
    args.output_path = os.path.join(args.output_dir, "story_line.txt")
    print_binary_dataset(args, all_story_line_dataset)            


def cluster_interface(args,news_list):
    news_vec = get_vectors_interface(method=args.encode,news_list = news_list, seq_len=512)
    y_pred = cluster_process(args, news_vec)
    #print('y_pred',y_pred)
    return y_pred

def story_cluster(args):
    files = os.listdir(args.input_dir)
    for f in files:
        #读取数据
        news_list = read_news_list(os.path.join(args.input_dir, f))
        #得到向量
        news_vec = get_vectors_interface(method=args.encode,news_list = news_list, seq_len=512)
        #开始聚类
        y_pred = cluster_process(args, news_vec)
        # 打印聚类结果
        args.output_path = os.path.join(args.output_dir, f)
        print_stories(args, y_pred, news_list)


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'story':
        if  os.path.exists(args.output_dir):
            os.system('rm -r {}'.format(args.output_dir))
        os.system('mkdir {}'.format(args.output_dir))        
        story_cluster(args)

    elif args.type == 'event':
        if  os.path.exists(args.output_dir):
            os.system('rm -r {}'.format(args.output_dir))
        os.system('mkdir {}'.format(args.output_dir))
        event_cluster(args)

    #normal_cluster(args)