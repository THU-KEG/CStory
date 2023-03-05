#from  RunClusters import *
import os
from news_vector import get_vectors_interface
import json
from collections import defaultdict
import argparse
import sys
sys.path.append('/data/skj/information_flow/model_bm25')
from CalculateScore import BM25
from RunClusters import cluster_interface
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='your input file')
    parser.add_argument( "--input_dir", help="your input directory")
    parser.add_argument("-e", "--encode", default='sentence_transformer')
    parser.add_argument("-c", "--cluster", default='SINGLEPASS')
    parser.add_argument("-s", "--story_file", help = "your output directory", default='/data/skj/information_flow/data_unprocessed/real_world_data/2019-11-15-2019-11-16.json')
    parser.add_argument("-k", "--event_dir")
    parser.add_argument("-n", "--num", help = "pre-defined cluster numbers", default=2)
    parser.add_argument("-t", "--threshold", help = "single pass threshold to split a cluster", default=0.7)
    parser.add_argument("-p", "--point", help = "min num points in a  cluster for dbscan", default=2)
    parser.add_argument("-r", "--radius", help = "radius of cluster for dbscan", default=0.7)
    parser.add_argument("--entity_number", help="entity number to identify two news", default=2)
    parser.add_argument("--ratio", help='Positive-negative sample ratio', default=2)
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

def cluster_to_events(args, news):
    args.threshold = 0.85
    #args.radius = 0.7
    events = defaultdict(list)
    labels = cluster_interface(args, news)
    for i, label in enumerate(labels):
        events[label].append(news[i])
    return events

def cluster_to_stories(args, news):
    args.threshold = 0.6
    #args.radius = 0.7
    stories = defaultdict(list)
    print(len(news))
    print(news[1])
    labels = cluster_interface(args, news)
    #print(labels)
    for i, label in enumerate(labels):
        stories[label].append(news[i])
    #print('events', stories)
    #print('kk',stories)
    filter_stories = defaultdict(list)
    for key in stories:
        #if len(stories[key]) > 1:
        filter_stories[key] = stories[key]
    return filter_stories


def select_important_events(args):
    filename = args.input_file
    #filenames = os.listdir(story_dir)
    #for filename in filenames:
    #file_path = os.path.join(story_dir, filename)
    news = json.load(open(filename))
    #news = news[:10000]
    print(len(news))
    #print(news[0])
    event_dict = cluster_to_events(args, news)
    print(len(event_dict))
    array = [(key, len(event_dict[key])) for key in event_dict]
    label2num = sorted(array,key=lambda x:x[1])
    
    indexs = [i[0] for i in label2num[:10000]]
    #print_to_file(indexs, event_dict, news)
    important_news = []
    for index in indexs:
        important_news.append(event_dict[index][0])
    print("important_news", len(important_news))
    w_f = open('tmp.json','w')
    json.dump(important_news, w_f, ensure_ascii=False)

# def merge_events(events):
#     for i in range(len(events)):
#         for j in range(i+1, len(events)):
            


if __name__ == '__main__':
    args = parse()
    #BM25_model = BM25()
    select_important_events(args)
    # for story in stories_dict.values():
    #     #print(story)
    #     #print(len(story))
    #     events_dict = cluster_to_events(args, story)

    #     #print(events_dict)
    #     events_list = [events_dict[i][0] for i in events_dict.keys()]
    #     if len(events_list) <= 1:
    #         continue
    #     #print(len(events_list))
    #     BM25_model.read_news_list(events_list)
    #     BM25_model.calculate_BM25()
    #     BM25_model.print()
    #     print('*********')
    


            