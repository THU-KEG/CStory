#from NewsMiner.newsminerPersoneventNew import *
#from Logger import Logger
from numpy import array
import json
import numpy as np 
def SINGLEPASS(all_vector, threshold):
    #logger.info("clustering started!")
    number_DocsinTopic = {}  # ，用于计算类的中心
    topic2vec = None
    DocumentinTopic = {}
    #all_vector, labels = get_vectors_interface(method='sentence_transformer', news_list = infile, seq_len = 128)
    #all_vector, labels = get_vectors_interface(method='skip_gram', news_list = infile)
    for count in range(all_vector.shape[0]):
        narry = [all_vector[count]]
        #all_vector.append(narry)
        # 一篇文档的vector计算完毕，开始计算与之前topic的余弦相似度进行合并
        dict_similarity = {}
        if count == 0:  # 第一篇文档，直接归为第一个topic
            # topic = "topic:0"
            topic = 0
            DocumentinTopic[topic] = [count]  # 将该文档的的标号归到该topic下
            topic2vec = narry
            number_DocsinTopic[topic] = 1.0
        else:  # 需要与前面的topic进行余弦相似度的比较
            dv = np.dot(topic2vec, np.transpose(narry))
            nv = np.linalg.norm(topic2vec, axis=1,
                                keepdims=True) * np.linalg.norm(narry, keepdims=True) + 0.01
            sims = dv / nv
            sim, topic = max((v[0], i) for i, v in enumerate(sims))
            if sim > threshold:  # 余弦相似度阈值
                # 找到同一个topic进行合并
                DocumentinTopic[topic].append(count)
                topic2vec[topic] = (topic2vec[topic] * number_DocsinTopic[topic] + narry[0]) / (
                    number_DocsinTopic[topic] + 1.0)
                number_DocsinTopic[topic] += 1.0
            else:  # 新的topic
                topic = len(topic2vec)
                DocumentinTopic[topic] = [count]
                topic2vec = np.concatenate((topic2vec, narry), axis=0)
                number_DocsinTopic[topic] = 1.0
    # 输出事件聚类中心向量
    return DocumentinTopic
