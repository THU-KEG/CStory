import json
import numpy as np 
from numpy import linalg as LA
from collections import defaultdict
count = 1
def get_simi(text_vector):
    length = len(text_vector)
    if length <= 1:
        return None
    similarity = np.zeros([length, length])
    norm_factor = np.ones(length)
    for i in range(length):
        norm_factor[i] = LA.norm(text_vector[i], 2)
    for i in range(length):
        for j in range(length):
            similarity[i][j] = np.sum(text_vector[i] * text_vector[j]) / norm_factor[i] / norm_factor[j]
    #print(similarity)
    return similarity

def dfs(i, core_labels, adjacent_matrix, count):
    for point in adjacent_matrix[i]:
        if core_labels[point] == 0:
            core_labels[point] = count
            dfs(point, core_labels, adjacent_matrix, count)
    

def second(array):
    if array.size == 0:
        return None
    first ,second = array[0], array[0]
    for element in array:
        if element > first:
            second = first
            first = element
        elif element > second:
            second = element
    return second

def recursive_method(similarity, labels, radius, points):
    global count
    n_samples = len(similarity)
    if n_samples <= points or radius >= 0.90:
        labels[:] = count
        count += 1
        return labels
    
    adjacent_matrix = defaultdict(list)

    core_samples = np.argwhere([np.sum(similarity[i] > radius) >= points for i in range(n_samples)]).squeeze(1)

    #print(core_samples)
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
        if min_simi < 0.7:
            cluster_label = np.full(len(sub_sample), count)
            origin_count = count
            count += 1
            recursive_method(sub_similarity, cluster_label, radius + 0.01, points)
            sub_core_samples = []
            for index, label in enumerate(cluster_label):
                if label != origin_count:
                    sub_core_samples.append(index)
            sub_core_samples = np.array(sub_core_samples)

            for index in range(len(cluster_label)):
                if label == origin_count:
                    print(index)
                    print(sub_core_samples)
                    nearest_neighbor =  np.argmax(sub_similarity[index][sub_core_samples])
                    cluster_label[index] = cluster_label[nearest_neighbor]
                core_labels[value[index]] = cluster_label[index]            

    for index, label in enumerate(core_labels):
        labels[core_samples[index]] = label
    
    for index in range(n_samples):
        if labels[index] == 0:
            max_simi_index = np.argmax(similarity[index][core_samples])
            if similarity[index][max_simi_index] > 0.7:
                labels[index] = labels[max_simi_index]
            else:
                labels[index] = count
                count += 1
    return labels

def NDBSCAN(text_vector, radius, points):
    text_vector = np.array(text_vector)
    similarity = get_simi(text_vector)
    labels = np.full(len(similarity), 0)
    labels = recursive_method(similarity, labels ,radius, points)
    return labels

def main():
    news_fp = open('test_data/trump1.json')
    news_list = json.load(news_fp)

    text = []
    for news in news_list:
        text.append(news['title'] + news['content'])
    news_vec = get_vectors_interface(method='sentence_transformer', text_list=text, seq_len=512)
    labels = NDBSCAN(news_vec, 0.7, 2)
    
    clusters = defaultdict(list)
    for index,label in enumerate(labels):
        clusters[label].append(index)
    
    cluster_count = 0
    for key, value in sorted(clusters.items(), key=lambda x:len(x[1]), reverse=True):
        if key == 0:
            continue 
        print("cluster" + str(cluster_count))
        cluster_count += 1
        for index in value:
            print(news_list[index]['title'])
        print()

