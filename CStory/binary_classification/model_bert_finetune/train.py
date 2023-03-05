from transformers import BertTokenizer
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import BertForMaskedLM
import os
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
	parser.add_argument("--template_type", type=str, default='base')
	parser.add_argument("--early_stop", type=int, default=20)
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
	parser.add_argument("--decay_rate", type=float, default=0.98)
	parser.add_argument("--weight_decay", type=float, default=0.0005)
	
	# lama configuration
	parser.add_argument("--only_evaluate", type=bool, default=False)
	parser.add_argument("--use_original_template", type=bool, default=False)
	parser.add_argument("--use_lm_finetune", type=bool, default=True)
	#prompt finetune
	parser.add_argument("--manner", type=str, default='finetune')
	
	
	parser.add_argument("--lstm_dropout", type=float, default=0.0)
	parser.add_argument("--train_data_path", type=str)
	parser.add_argument("--dev_data_path", type=str)
	parser.add_argument("--test_data_path", type=str)
	parser.add_argument("--out_dir", type=str)

	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--epoch", type=int, default=10)
	parser.add_argument("--no_cuda", type=bool, default=False)
	parser.add_argument('--max_length', type=int, default=512)
	#dataset format
	parser.add_argument('--dataset_format', type=str, default='normal')
	
	args = parser.parse_args()
	args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
	args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
	set_seed(args)
	return args
	
	
class Trainer(object):
	def __init__(self, args):
		self.args = args
		self.device = 'cuda'
		if  self.args.use_original_template and (not self.args.use_lm_finetune)	and (not self.args.only_evaluate):
			raise RuntimeError("""If use args.use_original_template is True, 
   			either args.use_lm_finetune or args.only_evaluate should be True.""")

		self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, use_fast=False)	
		self.train_set = TsvDataset(file_path=self.args.train_data_path)
		self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True) 
		self.dev_set = TsvDataset(file_path=self.args.dev_data_path)
		self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size)
		self.test_set = TsvDataset(file_path=self.args.test_data_path)
		self.test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size)
		# if self.args.manner == 'prompt':
		# 	self.model = PromptModel(self.args).to(self.device)
			#self.model = nn.DataParallel(PromptModel(self.args).to(self.device))
		if self.args.manner == 'finetune':
			self.model = FinetuneModel(self.args).to(self.device)
			#self.model = nn.DataParallel(FinetuneModel(self.args).to(self.device))
	def evaluate(self, epoch_idx, evaluate_type):
		self.model.eval()
		if evaluate_type == 'Test':
			loader = self.test_loader
			dataset = self.test_set
		elif evaluate_type == 'Train':
			loader = self.train_loader
			dataset = self.train_set
		else:
			loader = self.dev_loader
			dataset = self.dev_set

		with  torch.no_grad():
			self.model.eval()
			hit1, loss, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive = 0,0,0,0,0,0
			for data in loader:
				_loss, _hit1, _true_positive_in_pred, _pred_positive, _recall_positive_in_label, _label_positive = self.model(data)		
				hit1 += _hit1
				true_positive_in_pred += _true_positive_in_pred
				pred_positive += _pred_positive
				recall_positive_in_label += _recall_positive_in_label
				label_positive += _label_positive
				loss += _loss.item()

			precision = true_positive_in_pred / (pred_positive + 1e-6) 
			recall = recall_positive_in_label / (label_positive + + 1e-6)
			f1 = 2 * precision * recall /(precision + recall + 1e-6)				
			hit1 /= len(dataset)
			print("Epoch {} Loss: {} Hit@1:{} precision:{} recall:{} f1:{}".format(epoch_idx,
                                                          loss / len(dataset), hit1, precision, recall, f1))

		return loss,hit1, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive

	def get_save_path(self):
		return join(self.args.out_dir, self.args.template_type)

	def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1, train_hit1):
		ckpt_name = "epoch_{}_dev_{}_test_{}_train_{}.ckpt".format( epoch_idx, round(dev_hit1 * 100, 4),round(test_hit1 * 100, 4), round(train_hit1 * 100, 4))

		if self.args.manner == 'prompt':
			return {'embedding': self.model.prompt_encoder.state_dict(),'pretrained_embedding': self.model.state_dict(),
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
		'train_hit@1': train_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.datetime.now(),
                'args': self.args}
		elif self.args.manner == 'finetune':
			return {'pretrained_embedding': self.model.state_dict(),
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
		'train_hit@1': train_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.datetime.now(),
                'args': self.args}


	def save(self, best_ckpt):
		ckpt_name = best_ckpt['ckpt_name']
		path = self.get_save_path()
		os.makedirs(path, exist_ok=True)

		ckpt_path_file = open(os.path.dirname(os.path.abspath(__file__))+'/'+'path.txt','w')
		ckpt_path = join(path, ckpt_name)
		ckpt_path_file.write(ckpt_path)
		torch.save(best_ckpt, ckpt_path)

	def train(self):
		best_dev, early_stop, has_adjusted = 0, 0, True
		best_ckpt = None
		params = []
		if self.args.template_type == 'p_tuning':	
			params.append({'params': self.model.prompt_encoder.parameters()})

		if self.args.use_lm_finetune:
			params.append({'params': self.model.model.parameters(), 'lr': 5e-6})
		optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
		my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

		for epoch_idx in range(self.args.epoch):
			print('begin')
			if epoch_idx > -1:
				dev_loss, dev_hit1, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive = self.evaluate(epoch_idx, 'Dev')
				if epoch_idx == 0:
					test_loss, test_hit1, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive  = self.evaluate(epoch_idx, 'Test')

				if epoch_idx > 0 and (dev_hit1 >= best_dev) or self.args.only_evaluate:
					test_loss, test_hit1, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive = self.evaluate(epoch_idx, 'Test')
					train_loss, train_hit1, true_positive_in_pred, pred_positive, recall_positive_in_label, label_positive = self.evaluate(epoch_idx, 'Train')
					best_ckpt = self.get_checkpoint(epoch_idx, dev_hit1, test_hit1, train_hit1)
					early_stop = 0
					best_dev = dev_hit1
				else:
					early_stop += 1
					if early_stop >= self.args.early_stop:
						self.save(best_ckpt)
						return best_ckpt
			if self.args.only_evaluate:
				break

            		# run training
			hit1, num_of_samples = 0, 0
			tot_loss = 0
			self.model.train()
			for batch_idx, batch in tqdm(enumerate(self.train_loader)):
				loss, batch_hit1, _, _, _, _  = self.model(batch)
				hit1 += batch_hit1
				tot_loss += loss.item()
				num_of_samples += len(batch)

				loss.backward()
				torch.cuda.empty_cache()
				optimizer.step()
				torch.cuda.empty_cache()
				optimizer.zero_grad()
			my_lr_scheduler.step()

		self.save(best_ckpt)
		return best_ckpt

def main():
	args = construct_args()
	trainer = Trainer(args)
	trainer.train()

if __name__ == '__main__':
    main()
					
	
		
		
	

