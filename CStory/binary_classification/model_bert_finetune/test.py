from email.policy import default
from transformers import BertTokenizer
from transformers import AutoTokenizer
from transformers import BertModel
import os
import torch.nn as nn
import torch
from os.path import join, abspath, dirname
from dataset import TsvDataset
from tqdm import tqdm
import torch.utils.data as Data
from torch.utils.data import DataLoader
from model import FinetuneModel
import datetime
import numpy as np
import argparse

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def construct_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default='bert-base-chinese')
        parser.add_argument("--early_stop", type=int, default=20)
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
        parser.add_argument("--decay_rate", type=float, default=0.98)
        parser.add_argument("--weight_decay", type=float, default=0.0005)
        parser.add_argument("--ckpt_name", type=str, default='')
        parser.add_argument("--template_type", type=str, default='base')
        parser.add_argument("--load_path",type=str, default='output/story_line/base/epoch_9_dev_85.3503_test_81.6118_train_98.5176.ckpt')
        # lama configuration
        parser.add_argument("--only_evaluate", type=bool, default=False)
        parser.add_argument("--use_original_template", type=bool, default=False)
        parser.add_argument("--use_lm_finetune", type=bool, default=True)
        #prompt finetune
        parser.add_argument("--manner", type=str, default='finetune')


        parser.add_argument("--lstm_dropout", type=float, default=0.0)
        parser.add_argument("--test_data_path", type=str, default='/ceph/qbkg/martinshi/event/data/Dtuning/few_shot_64_train.tsv')
        parser.add_argument("--out_dir", type=str, default='/ceph/qbkg/martinshi/event/output_finetune')

        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--no_cuda", type=bool, default=False)
        parser.add_argument('--max_length', type=int, default=512)

        args = parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        set_seed(args)
        return args


class Tester(object):
        def __init__(self, args):
                self.args = args
                self.device = 'cuda'

                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast=False)
                self.test_set = TsvDataset(file_path=self.args.test_data_path)
                self.test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size)

                self.model = FinetuneModel(self.args).to(self.device)
        
        def evaluate(self, evaluate_type):
                self.model.eval()
                if evaluate_type == 'Test':
                        loader = self.test_loader
                        dataset = self.test_set

                with  torch.no_grad():
                        self.model.eval()
                        hit1, loss, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive = 0,0,0,0,0,0
                        all_logits = None
                        all_label = None
                        for data in loader:
                                _loss, _hit1, _true_positive_in_pred, _pred_positive, _recall_positive_in_label, _label_positive = self.model(data)	
                                hit1 += _hit1
                                true_positive_in_pred += _true_positive_in_pred
                                pred_positive += _pred_positive
                                recall_positive_in_label += _recall_positive_in_label
                                label_positive += _label_positive
                                loss += _loss.item()

                                logit = self.model.get_logits(data)
                                if all_logits == None:
                                        all_logits = logit
                                else:
                                        all_logits = torch.cat((all_logits, logit), dim= 0)
                                
                                if all_label == None:
                                        all_label = torch.LongTensor([int(i) for i in data['label']])
                                else:
                                        tmp_label = torch.LongTensor([int(i) for i in data['label']])
                                        all_label = torch.cat((all_label, tmp_label),dim=0)
                        
                        #print(all_logits)
                        #print(all_label)
                        #pred_labels = (all_logits[:,-1]>0.5).long()
                        #hit1 = int(torch.sum(pred_labels == all_label.cuda()))
                        precision = true_positive_in_pred / pred_positive
                        recall = recall_positive_in_label / label_positive
                        f1 = 2 * precision * recall /(precision + recall)				
                        hit1 /= len(dataset)
                        print("Loss: {} Hit@1:{} precision:{} recall:{} f1:{}".format(loss / len(dataset), hit1, precision, recall, f1))
        
                return loss/len(dataset), hit1, all_logits

        def get_save_path(self):
                return 'output/story_line/base/epoch_9_dev_85.3503_test_81.6118_train_98.5176.ckpt'


        def load(self):
                path = self.args.load_path
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['pretrained_embedding'])

        def test(self):
                self.load()
                test_loss, test_hit1, test_logits = self.evaluate( 'Test')
                #dev_loss, dev_hit1, dev_logits = self.evaluate('Dev')
                #train_loss, train_hit1, train_logits = self.evaluate('Train')
                test_list = test_logits.tolist()
                #dev_list = dev_logits.tolist()
                #train_list = train_logits.tolist()
                # test_f = open('test.txt','w')
                # #dev_f = open('dev.txt','w')
                # #train_f = open('train.txt','w')
                # for line in test_list:
                #          line = ' '.join([str(i) for i in line])
                #          test_f.write(line+'\n')
                # for line in dev_list:
                #          line = ' '.join([str(i) for i in line])
                #          dev_f.write(line + '\n')
                # for line in train_list:
                #          line = ' '.join([str(i) for i in line])
                #          train_f.write(line + '\n')
                 
                       
                #train_loss, train_hit1, train_logits = self.evaluate('Train')
                #print('train_loss:{} train_hit1:{}'.format(train_loss, train_hit1))

def main():
        args = construct_args()
        tester = Tester(args)
        tester.test()

if __name__ == '__main__':
        main()        	







