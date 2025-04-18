import torch
import numpy as np
import pdb
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import math

def load_tsp_data():
    data = torch.load('data/tsp/tsp_data.pt')
    return data

class TSPDataset(Dataset):
    def __init__(self, size=15, length = 1000, type = 'train',seed=1234):
        self.size = size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = load_tsp_data()
        self.input,self.label = self.data['input'],self.data['label']

        if type == 'train':
            self.input = self.input[:length]
            self.label = self.label[:length]
        elif type == 'test':
            self.input = self.input[length:]
            self.label = self.label[length:]

        self.input = self.input.to(device)
        self.label = self.label.to(device) 

    
    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        
        return dict(
            input = self.input[idx],
            label = self.label[idx]
        )



    
        
        

