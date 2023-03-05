import json
from posixpath import dirname
import pandas as  pd
import os
from collections import defaultdict
path = '/data/skj/data/ccl.tsv'
print(path)
path_array = path.split('/')
dir_path = '/'.join(path_array[:-1])
df = pd.read_csv(path,'\t')
#df =df.drop('Unnamed: 0',axis=1)



file = '/data/skj/information_flow/data_unprocessed/real_world_data/raw_2021-05-20_2021-06-02.json'

count = 0
def read_news_list(data_file):
    global count
    print(data_file)
    with open(data_file, 'r') as file:
        news_list = json.load(file)
    titles = set()
    res = []
    for news in news_list:
        if news['title'] not in titles:
            titles.add(news['title'])
            news['newsID'] = count
            count += 1
            res.append(news)
    #print("num of news:", len(res))
    return res

def find_news(news_list, news_id):
    for news in news_list:
        if news_id == news['newsID']:
            return news 
    return None

news_list = read_news_list(file)


news_dict = defaultdict(str)
for news in news_list:
    tmp = ''
    if news['title']:
        tmp += news['title']
    if news['content']:
        tmp += news['content']
    
    tmp = tmp.replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','')
    #print(news.keys())
    news_dict[tmp] = news['newsID']

print("DDDDDFFFFFF",len(df))
valid_num = 0

triple = pd.DataFrame(columns=('id_1','id_2','label'))

valid_news = []
valid_news_id = set()

for index,row in df.iterrows():
    seq1, seq2, label = row[0], row[1], row[2]
    #内容必须吻合
    if seq1 not in news_dict or seq2 not in news_dict:
        continue;
    valid_num += 1
    #寻找对应的新闻并且加入到valid_news队列中
    if news_dict[seq1] not in valid_news_id:
        candidate_news =  find_news(news_list,news_dict[seq1])
        if candidate_news:
            valid_news.append(candidate_news)
            valid_news_id.add(news_dict[seq1])

    if news_dict[seq2] not in valid_news_id:
        candidate_news =  find_news(news_list,news_dict[seq2])
        if candidate_news:
            valid_news.append(candidate_news)
            valid_news_id.add(news_dict[seq2])
    #生成id，id,label形式的三元组
    print({'id_1':news_dict[seq1],'id_2':news_dict[seq2],'label':label})
    triple.loc[valid_num]={'id_1':news_dict[seq1],'id_2':news_dict[seq2],'label':label}
    print("succeed once")

    
    #print(index)


print(valid_num)
json.dump(valid_news, open(dir_path+'/'+path_array[-1]+'_DATA.json','w'), ensure_ascii=False)
print('KKK',triple)
print(len(triple))
triple.to_csv(open(dir_path+'/'+path_array[-1]+'_DATA.csv','w'))
