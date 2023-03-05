import json
from torch.utils.data import Dataset
import pandas as pd
class TsvDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path,'\t')
        self.df = self.df.dropna(axis=0, how='any')
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        obj = dict(self.df.iloc[index])
        return obj
