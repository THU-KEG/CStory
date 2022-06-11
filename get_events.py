import json
from operator import ne
import numpy as np 
from numpy import linalg as LA
from collections import defaultdict
from news_vector import get_vectors_interface
import logging
import requests
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#handler = logging.FileHandler("Log/test2.log",mode='w')
#logger.addHandler(handler)
count = 1
sign = False

def get_simi(vec):
    print('vec', vec.shape)
    X_norm = LA.norm(vec,2,1)
    similarity = np.zeros((len(vec), len(vec)))  
    for i in range(len(vec)):
        column = np.sum(vec[i] * vec,1) / X_norm[i] / X_norm
        similarity[i][:] = column 
        #print(i)
    
    
    return similarity

def second(row):
    if len(row) == 0:
        return None
    first, second = row[0], row[0]
    for element in row:
        if element > first:
            second = first
            first = element
        elif element > second:
            second = element
    return second

def argsecond(row):
    if len(row) == 0:
        return None
    first, second = 0, 0
    for index,element in enumerate(row):
        if element > first:
            second = first        
            first = index
        elif element > second:
            second = index
    return second

def dfs(i, core_labels, adjacent_matrix, count):
    for point in adjacent_matrix[i]:
        if core_labels[point] == 0:
            core_labels[point] = count
            dfs(point, core_labels, adjacent_matrix, count)

def complete_cluster(similarity, radius):
    n_samples = len(similarity)
    min_simi = np.min(similarity)
    if n_samples <= 2 or min_simi > radius:
        return True
    return False

def convert_label_to_cluster(labels, origin_count):
    cluster = defaultdict(list)
    for index, label in enumerate(labels):
        cluster[label].append(index)
    solo_point = []
    for key in cluster:
        if len(cluster[key]) == 1 and key != origin_count:
            solo_point.append(cluster[key][0])
    return solo_point

def find_cluster(cluster_set, index):
    for key,item in cluster_set.items():
        if index in item:
            return item
    return []

def merge(similarity, index, cluster, radius):
    cluster = np.append(cluster, index)
    sub_simi = similarity[cluster,:][:,cluster]
    min_simi = np.min(sub_simi)
    if min_simi < radius:
        return False
    return True


def recursive_method(similarity, labels, radius, origin_radius, points):
    global count
    n_samples = len(similarity)
    if complete_cluster(similarity, origin_radius):
        labels[:] = count
        count += 1
        return labels
    
    adjacent_matrix = defaultdict(list)
    core_samples = np.argwhere([np.sum(similarity[i] > radius) >= points for i in range(n_samples)]).squeeze(1)
    
    n_core_samples = len(core_samples)
    if n_core_samples == 0:
        for index in range(n_samples):
            labels[index] = count
            count += 1

    for i in range(n_core_samples):
        for j in range(i, n_core_samples):
            if i != j and  similarity[core_samples[i]][core_samples[j]] > radius:
                adjacent_matrix[i].append(j)
                adjacent_matrix[j].append(i)

    core_labels = np.full(n_core_samples, 0)
    for i in range(n_core_samples):
        if core_labels[i] == 0:
            core_labels[i] = count
            dfs(i, core_labels, adjacent_matrix, count)
            count += 1

    core_cluster = defaultdict(list)
    for index,label in enumerate(core_labels):
        core_cluster[label].append(index)

    for key,value in core_cluster.items():
        sub_sample = core_samples[value]
        sub_similarity = similarity[sub_sample,:][:,sub_sample]
        min_simi = np.min(sub_similarity)
        print("min_simi",min_simi)
        if min_simi <= origin_radius:
            cluster_label = np.full(len(sub_sample), count)
            origin_count = count
            count += 1
            print(value)
            print(len(value))
            # next_radius = 1
            # for row in sub_similarity:
            #     next_radius = min(next_radius, second(row))
    
            recursive_method(sub_similarity, cluster_label, radius + 0.01, origin_radius, points)

            sub_core_samples = []
            sub_cluster = defaultdict(list)
            for index, label in enumerate(cluster_label):
                if label != origin_count:
                    sub_core_samples.append(index)
                sub_cluster[label].append(index)
            sub_core_samples = np.array(sub_core_samples)

            solo_point = convert_label_to_cluster(cluster_label, origin_count)
            
            for index in range(len(cluster_label)):
                if cluster_label[index] == origin_count:
                    nearest_neighbor =  np.argmax(sub_similarity[index][sub_core_samples])
                    nearest_cluster = find_cluster(sub_cluster, nearest_neighbor)
                    assert nearest_cluster != []
                    if merge(sub_similarity, index, nearest_cluster, origin_radius):
                        cluster_label[index] = cluster_label[nearest_neighbor]
                elif index in solo_point:
                    nearest_neighbor = argsecond(sub_similarity[index][sub_core_samples])
                    nearest_cluster = find_cluster(sub_cluster, nearest_neighbor)
                    if merge(sub_similarity, index, nearest_cluster, origin_radius):
                        cluster_label[index] = cluster_label[nearest_neighbor]
                core_labels[value[index]] = cluster_label[index]      

    for index, label in enumerate(core_labels):
        labels[core_samples[index]] = label
    
    for index in range(n_samples):
        if labels[index] == 0:
            max_simi_index = np.argmax(similarity[index][core_samples])
            if similarity[index][max_simi_index] > origin_radius:
                labels[index] = labels[max_simi_index]
            else:
                labels[index] = count
                count += 1
    return labels

def NDBSCAN(text_vector, radius, points, news_list):
    text_vector = np.array(text_vector)
    #similarity = get_simi(text_vector)
    #similarity = get_time_simi(text_vector, time)
    labels = np.full(len(similarity), 0)
    labels = recursive_method(similarity, labels ,radius, radius, points)   


    single_point = []
    # for i in range(len(similarity)):
    #     if np.sum(similarity[i]>0.7) >= 1:
    #         single_point.append(i)
    single_point = np.argwhere(np.array([np.sum(similarity[i] > 0.7) for i in range(len(similarity))])>=2).squeeze(1) 
    return labels
        
def find_key(cluster, nearest_neighbor):
    for key, value in cluster.items():
        for index in value:
            if index == nearest_neighbor:
                return key
    return None

def INDBSCAN_(radius, points, similarity):
    global count
    total_index = []
    cluster = defaultdict(list)
    for index in range(similarity.shape[0]):
        if len(total_index) == 0:
            total_index.append(index)
            cluster[count].append(index)
            count += 1
            continue

        nearest_neighbor = np.argmax(similarity[index][total_index])
        total_index.append(index)

        if similarity[index][nearest_neighbor] > radius:
            nearest_key = find_key(cluster, nearest_neighbor)

            if nearest_key == None:
                print("找不到最近的邻居")
                return None
                
            cluster[nearest_key].append(index)
            candidate = cluster[nearest_key]

            labels = np.full(len(candidate), 0)

            recursive_method(similarity[candidate,:][:,candidate], labels, radius, radius, points)

            cluster.pop(nearest_key)

                
            for label_index,label in enumerate(labels):
                cluster[label].append(candidate[label_index])
                
        else:
            cluster[count].append(index)
            count += 1    

    labels = np.full(len(similarity), 0)
    for key,value in cluster.items():
        labels[value] = key

    return labels

def INDBSCAN(text_vector, radius, points, sen_bert_model=None, news_list=None):
    global count
    cluster = defaultdict(list)    
    text_vector = np.array(text_vector)
    similarity = get_simi(text_vector)
    #keyword_similarity = get_keyword_similarity(sen_bert_model, news_list)
    #alpha = 1
    #assert similarity.shape == keyword_similarity.shape
    #similarity = alpha * similarity + (1 - alpha) * keyword_similarity
    #similarity = get_time_simi(text_vector, time_list)
    total_index = []
    
    for index,vector in enumerate(text_vector):
        if len(total_index) == 0:
            total_index.append(index)
            cluster[count].append(index)
            count += 1
            continue

        nearest_neighbor = np.argmax(similarity[index][total_index])
        total_index.append(index)

        if similarity[index][nearest_neighbor] > radius:
            nearest_key = find_key(cluster, nearest_neighbor)

            if nearest_key == None:
                print("找不到最近的邻居")
                return None
                
            cluster[nearest_key].append(index)
            candidate = cluster[nearest_key]

            labels = np.full(len(candidate), 0)

            recursive_method(similarity[candidate,:][:,candidate], labels, radius, radius, points)

            cluster.pop(nearest_key)

                
            for label_index,label in enumerate(labels):
                cluster[label].append(candidate[label_index])
                
        else:
            cluster[count].append(index)
            count += 1    

    labels = np.full(len(similarity), 0)
    for key,value in cluster.items():
        labels[value] = key

    return labels

def main():
    news_fp = open('/data/skj/information_flow/data_unprocessed/specific_domain_data/20210916.json-20210922.json')
    news_list = json.load(news_fp)

    news_vec = get_vectors_interface(method='sentence_transformer', news_list=news_list, seq_len=512)
    #print("FFFFF", news_vec)
    labels = INDBSCAN(news_vec[0], 0.75, 2)
    
    clusters = defaultdict(list)
    for index,label in enumerate(labels):
        clusters[label].append(index)
    
    cluster_count = 0
    w_file = open('latest.txt', 'w')
    for key, value in sorted(clusters.items(), key=lambda x:len(x[1]), reverse=True):
        if key == 0:
            continue 
        print("cluster" + str(cluster_count)+ "  "+ str(len(value)))
        w_file.write('cluster' + str(cluster_count) + str(len(value)) + '\n')
        cluster_count += 1
        for index in value:
            print(news_list[index]['title'])
            w_file.write(news_list[index]['title']+"\n")
        print()
        w_file.write('\n')

main()