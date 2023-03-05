import numpy as np
from numpy import linalg as LA
from collections import defaultdict
#import logging
#logging.basicConfig(level=logging.INFO)
#logger =logging.getLogger(__name__)
#handler = logging.FileHandler("Log/test1.log",mode='w')
#logger.addHandler(handler)

#设置类别的标签，为了不重复，使用了cycl*10000
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

def find_component(v, adjacency, component, all_indices,count, origin_index):
    """
    dfs to find connected component for a graph
    """
    component[v] = count
    all_indices.append(v)
    for u in adjacency[v]:
        if component[u] == origin_index:
            find_component(u, adjacency, component,all_indices,count, origin_index)
    return 


def get_story_sample(similarity, eps, min_samples,count, news_list):
    news_title = []
    for news in news_list:
        news_title.append(news['title'])
    news_title = np.array(news_title)
    #min_samples = min_samples_list[1]
    #一共的节点个数
    print('essssp', eps)
    n_samples = similarity.shape[0]
    #每个节点的邻居个数
    n_neighbors = np.sum(similarity >= eps,axis=1)
    #print("NEIGHBOR", n_neighbors)
    #获得核心节点的位置
    core_samples = np.argwhere(n_neighbors >= min_samples).squeeze(axis=1)
    #print("core_samples", core_samples)
    #获得核心节点的个数
    n_core_samples = core_samples.shape[0]
    #print('core_samples',len(core_samples), core_samples)
    #获取核心节点的邻接矩阵
    adjacency = defaultdict(list)
    for i in range(n_core_samples):
        idx, next_samples = core_samples[i], core_samples[i:]
        dist =  similarity[idx , next_samples] 
        neighbors = np.argwhere(dist >= eps).squeeze(axis=1)
        for neighbor_idx in next_samples[neighbors]:
            adjacency[idx].append(neighbor_idx)
            adjacency[neighbor_idx].append(idx)
    
    #初始化label
    labels = np.full(n_samples, -1)
    print("本次迭代长度",len(labels))
    # 对于每一个核心节点，通过dfs进行聚类
    aaa = 0
    for idx in core_samples:
        #print("pre",idx)
        if labels[idx] == -1:
            aaa += 1
            #print("IDX",idx)
            new_indices = []
            find_component(idx, adjacency, labels, new_indices,count, -1)
            count += 1
            new_similarity = similarity[new_indices,:][:,new_indices]
            #如果超过我们设定的某一个限制，证明这是一个story，进行event的识别
            if (len(new_similarity) >= min_samples):
                next_title = news_title[new_indices]
                #logger.info("aaa:{},len(news_title),{}".format(aaa,len(new_indices)))
                new_label,count = get_event_sample(similarity=new_similarity, eps=eps + 0.01,count=count,news_title=next_title, origin_eps=eps)
                print("len(news_title)",len(news_title))
                #print(new_label)
                for index,label in enumerate(new_label):
                    labels[new_indices[index]] = label
                #print(labels)
                #logger.info(labels)

    #将不是核心点的节点按照距离大小强制分入类中
    count = add_edge_dot(similarity=similarity,labels=labels, core_samples=core_samples, eps=eps,count=count,origin_eps=eps)
    return labels, count 

def get_event_sample(similarity, eps,count, news_title,origin_eps):
    min_samples = 3
    origin_index = count - 1
    #一共的节点个数
    n_samples = similarity.shape[0]
    #初始化label为原始的label
    labels = np.full(n_samples,origin_index)
    #print("LLLL",labels)
    if n_samples == 2 or eps >= 0.90:
        return labels, count
    #每个节点的邻居个数
    n_neighbors = np.sum(similarity >= eps,axis=1)
    #print("NNNN", n_neighbors)
    #获得核心节点的位置
    core_samples = np.argwhere(n_neighbors >= min_samples).squeeze(axis=1)
    #print('YYYY', core_samples.size)
    #如果不存在核心节点，那就没必要继续了
    if core_samples.size == 0:
        #print("KKKKK", labels)
        return labels,count
    #获得核心节点的个数
    n_core_samples = core_samples.shape[0]
    
    #获取核心节点的邻接矩阵
    adjacency = defaultdict(list)
    for i in range(n_core_samples):
        idx, next_samples = core_samples[i], core_samples[i:]
        dist =  similarity[idx , next_samples] 
        neighbors = np.argwhere(dist >= eps).squeeze(axis=1)
        for neighbor_idx in next_samples[neighbors]:
            adjacency[idx].append(neighbor_idx)
            adjacency[neighbor_idx].append(idx)  

    # 对于每一个核心节点，通过dfs进行聚类
    for idx in core_samples:
        #print("pre",idx)
        if labels[idx] == origin_index:
            #print("IDX",idx)
            new_indices = []
            find_component(idx, adjacency, labels, new_indices,count, origin_index)
            count += 1
            # print("EVENT_CORE_SAMPLE", core_samples)  
            # print("indices", new_indices)
            
            new_similarity = similarity[new_indices,:][:,new_indices]
            #如果超过我们设定的某一个限制，证明这是一个story，进行event的识别
            if (len(new_similarity) >= min_samples):
                next_title = news_title[new_indices]
                new_label,count = get_event_sample(new_similarity,eps+0.01,count, next_title,origin_eps)
                logger.info("eps:{:.4f},event_label:{}, next_title:{}".format(eps+0.01,new_label,next_title))
                for index,label in enumerate(new_label):
                    labels[new_indices[index]] = label
    count = add_edge_dot(similarity=similarity,labels=labels, core_samples=core_samples, eps=eps, count=count,origin_eps=origin_eps)
    return labels,count
    

def add_edge_dot(similarity,  labels, core_samples, eps, count, origin_eps):
    #获取总节点的个数
    n_samples = similarity.shape[0]
    #获取核心节点的个数
    n_core_samples = core_samples.shape[0]
    if n_core_samples == 0 :
        labels[:] = -2
        return
    for i in range(n_samples):
        if i in core_samples:
            continue
        
        dist_core = similarity[i, core_samples]
        max_core = np.max(dist_core)
        if max_core >= origin_eps:
            most_proper_core = core_samples[np.argmax(dist_core)]
            labels[i] = labels[most_proper_core]
        else:
            labels[i] = -2
    return count

def RDBSCAN(eps,news_vec, min_samples, news_list):
    similarity = init_vector(news_vec)
    print(similarity)
    print("EPS",eps)
    print("min_samples", min_samples)
    labels,count = get_story_sample(similarity=similarity, min_samples=min_samples, eps=eps, count=0, news_list=news_list)
    print(np.sum(labels==-2))
    logger.info("等于-1的有{}".format(np.sum(labels==-1)))
    eps -= 0.01
    while (eps >= 0.7):
        max_index = np.max(labels)
        noise_sample = []
        for index,label in enumerate(labels):
            if label == -2:
                noise_sample.append(index)
        if noise_sample:
            logger.info(labels)
            noise_sample = np.array(noise_sample)
            #print("LABELS",labels)
            print("NOISE_sample", noise_sample)
            noise_similarity = similarity[noise_sample,:][:,noise_sample]
            noise_labels,count = get_story_sample(similarity=noise_similarity, min_samples=min_samples, eps=eps,count=max_index+1, news_list=np.array(news_list)[noise_sample])
            for index,noise_label in enumerate(noise_labels):
                #print("BBBB", noise_labels, type(noise_labels))
                labels[noise_sample[index]] = noise_labels[index]
            print("挑选出来的新的新闻,",np.sum(noise_labels!=-2))
            print("一共的新闻", len(noise_labels))
            print('NOISE',len(noise_sample))
            eps -= 0.01
        else:
            break
    clusters = set()
    for d_index,label in enumerate(labels):
        clusters.add(labels[d_index])
    print("分类的数量", len(clusters))
    logger.info("一共的数量:{}".format(len(labels)))
    for index,label in enumerate(labels):
        if label == -2:
            labels[index] = count
            count += 1
    logger.info("有效的数量:{}".format(len(labels[labels != -2])))
    print(labels)
    #print(similarity)
    return labels
