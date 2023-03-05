import numpy as np
from numpy import linalg as LA
from collections import defaultdict


def init_vector(vec):
    # time complexity :n2
    X_norm = LA.norm(vec,2,1)
    similarity = np.zeros((len(vec), len(vec)))  
    for i in range(len(vec)):
        #column = LA.norm(vec[i]-vec,axis=1)
        column = np.sum(vec[i] * vec,1) / X_norm[i] / X_norm
        similarity[i][:] = column 
        #print(i)
    return similarity

def find_component(v, component_id, adjacency, component, all_indices):
    """
    dfs to find connected component for a graph
    """
    component[v] = component_id
    all_indices.append(v)
    for u in adjacency[v]:
        if component[u] == -1:
            find_component(u, component_id, adjacency, component,all_indices)
    return


def get_core_sample(core_samples,similarity,eps,labels):
    n_samples = len(similarity[0])
    #print('core_samples',len(core_samples), core_samples)
    n_core_samples = core_samples.shape[0]
    # build graph, core neighbors
    adjacency = defaultdict(list)
    for i in range(n_core_samples):
        idx, next_samples = core_samples[i], core_samples[i:]
        dist =  similarity[idx , next_samples] 
        neighbors = np.argwhere(dist >= eps).squeeze(axis=1)
        for neighbor_idx in next_samples[neighbors]:
            adjacency[idx].append(neighbor_idx)
            adjacency[neighbor_idx].append(idx)
    # find connected component
    n_components = 0
    for idx in core_samples:
        if labels[idx] == -1:
            all_indices = []
            find_component(idx, n_components, adjacency, labels,all_indices)
            print(all_indices)
            n_components += 1
    #print("BBBB",labels)

def calculate(eps, min_samples, similarity):
    n_samples = similarity.shape[0]
    n_neighbors = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        n_neighbors[i] = np.sum(similarity[i] >= eps)
    is_core = n_neighbors >= min_samples
    core_samples = np.argwhere(is_core).squeeze(axis=1)
    n_core_samples = core_samples.shape[0]
    labels = np.full(n_samples,-1)

    get_core_sample(core_samples,similarity,eps,labels)

    #print("labels",labels)
    label2num = defaultdict(list)
    for index,label in enumerate(labels):
        if label != -1:
            label2num[label].append(index)
           
    core_label = [x[0] for x in sorted(label2num.items(), key=lambda x: len(x[1]), reverse=True)]
    #print("CORE_label",core_label)

    #solid_eps = 0.8
    # assign each non-core sample to its nearest component (component of its nearest core sample)
    for i in range(n_samples):
        if is_core[i]:
            continue
        if n_core_samples == 0:
            break
        dist_core = similarity[i, core_samples]
        #candidate = similarity[i, core_samples]
        #dist_core = np.argwhere(candidate > eps).squeeze()
        max_core = np.max(dist_core)
        if max_core >= eps:
            most_proper_core = core_samples[np.argmax(dist_core)]
            labels[i] = labels[most_proper_core]

    print("分类的数量", len(labels[labels != -1]))
    max_count = np.max(labels)
    #labels = labels.tolist()
    for index,label in enumerate(labels):
        if label == -1:
            max_count += 1
            #print(index, max_count)
            labels[index] = max_count
    #print("dbscan的label", labels)
    return labels
def DBSCAN(eps,min_samples,news_vec):
    """
    :param X: (n_samples, n_features)
    :param eps: the radius to count neighboring samples
    :param min_samples: min number of neighboring samples
                        to be a core sample (including itself)
    :return: cluster labels for each sample
             -1 means noise sample
    """
    similarity = init_vector(news_vec)
    return calculate(eps, min_samples, similarity)
def DBSCAN_(eps, min_samples, similarity):
    return calculate(eps, min_samples, similarity)
