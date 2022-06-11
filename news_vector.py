import numpy as np
import jieba
import numpy.linalg as LA
import json
import torch
from sentence_transformers import SentenceTransformer,util

sen_bert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
def get_vectors_interface(method,news_list, stopwords=None, word2id=None, model_path=None,seq_len=None,batch_size=None,):
    news_vec = None
    label = None
    
    if method == 'sentence_transformer':
        news_vec = get_sentence_transformer(news_list,seq_len)
        return news_vec

    return news_vec, label

def get_sentence_transformer(news_list,seq_len):
    sen_bert_model.max_seq_len = seq_len
    contents = []
    mark = []
    for index,news in enumerate(news_list):
        #content = news['title']
        content = news['title'] + news['content']
        #content.split('\n')
        content = content if len(content) < 256 else content[:256]
        contents.append(content)
        # pieces = int(np.ceil(len(content)/510.0))
        # for piece in range(pieces):
        #     chunk = content[piece * 510 : (piece+1) * 510]
        #     contents.append(chunk)
        #     mark.append(index)
        # # if len(content) > (seq_len - 2):
        # #     content = content[:seq_len - 2]
    res = sen_bert_model.encode(contents)
    #res = []
    #nums = []
    # for order,sentence_vector in enumerate(vector):
    #     if mark[order] > len(res) - 1:
    #         res.append(sentence_vector)
    #         nums.append(1)

    #     elif mark[order] == len(res) - 1:
    #         res[-1] += sentence_vector
    #         nums[-1] += 1
    #     else:
    #         print("出错了")

    #for order in range(len(res)):
    #    res[order] /= nums[order]
    assert(len(news_list) == len(res))
    res = np.array(res)
    return res

# def get_word2vec(news_list, stopwords, word2id, model_path = '/data1/skj/Glove/abc.bin'):
#     news_vec = []
#     label = []
#     def doc2vec(content_seg):
#         vec = torch.zeros(300)
#         for word in content_seg:
#             if word in model:
#                 vec += model[word]
#         return vec
                
#     for news in news_list:
#         content = news['title'] + news['content']
#         content_seg = [word for word in jieba.lcut(content, cut_all=False) \
#                        if not word in stopwords]
#         vec = doc2vec(content_seg)
#         news_vec.append(vec)
#         if 'label' in news.keys():
#             label.append(news['label'])

#     news_vec = np.stack(news_vec, axis=0)
#     return news_vec, label


# def get_bertvec(news_list,seq_len, batch_size):
#     bert_model = BertModel.from_pretrained('/data/skj/newsminer/chinese-openie/bert_chinese')
#     bert_tokenizer = BertTokenizer.from_pretrained('/data/skj/newsminer/chinese-openie/bert_chinese')
#     bert_model.eval()
#     news_vec = []
#     labels = []
#     batch_vec = torch.LongTensor(batch_size, seq_len)
#     for index, news in enumerate(news_list):
#         if 'label' in news.keys():
#             labels.append(news['label'])
#         content = news['title'] + news['content']
#         if len(content) > (seq_len - 2):
#             content = content[:seq_len - 2]
#         ids = bert_tokenizer.encode(content)
#         if len(ids) < seq_len:
#             ids += [0] * (seq_len-len(ids))
#         batch_vec[index % batch_size]=torch.tensor(ids)
#         sentences, maxpooling = bert_model(batch_vec)
#         if  (index+1) % batch_size ==0:
#             #print(index)
#             #print(batch_vec.shape)
#             #print("YYY",sentences.shape)
#             for i in range(len(maxpooling)):
#                 #print(torch.mean(sente))
#                 news_vec.append(torch.mean(sentences[i],0).detach().numpy())
          
#         if index == len(news_list) - 1:
#             for i in range(index % batch_size):
#                 news_vec.append((torch.mean(sentences[i],0)).detach().numpy())
#             break
#     return news_vec,labels     
        


# def get_skip_gram(news_list, stopwords, word2id):
#     vocab = word2id

#     N_CATEGORIES = 2500
#     def doc2vec(doc):
#         vec = np.zeros(N_CATEGORIES, dtype=np.int32)
#         for word in doc:
#             if word in vocab:
#                 vec[word2id[word]] += 1
#         return vec

#     news_vec = []
#     label = []
#     for news in news_list:
#         content = news['title'] + news['content']
#         content_seg = [word for word in jieba.lcut(content, cut_all=False) \
#                        if not word in stopwords]
#         vec = doc2vec(content_seg)
#         news_vec.append(vec)
#         if 'label' in news.keys():
#             label.append(news['label'])

#     news_vec = np.stack(news_vec, axis=0)

#     return news_vec, label
