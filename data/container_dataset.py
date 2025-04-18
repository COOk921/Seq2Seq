import torch
import numpy as np
import pdb
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import math
from data.process import deal_container_data

class ContainerDataset(Dataset):
    def __init__(self,size = 8, type = 'train', seed = 1234):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = deal_container_data()
        self.data = self.data.to(device)
     
        self.num_samples = self.data.shape[0] // size
        
        if self.num_samples * size != self.data.shape[0]:
            self.data = self.data[:self.num_samples * size, :]

        self.data = self.data.view(self.num_samples, size, self.data.shape[1])
        
        torch.manual_seed(seed)
        self.data = self.data[:, torch.randperm(self.data.size(1)), :]

        # 8:2
        length = int(self.num_samples * 0.8) 
        if type == 'train':
            self.data = self.data[:length,:,:]
        else:
            self.data = self.data[length:,:,:]
        
        self.label = self.data[:,:,-1]
        self.input = self.data[:,:,:-1]
        
        self.label = torch.argsort(self.label, dim=-1)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        data = dict(
            input = self.input[idx],
            label = self.label[idx]
        )

        return data

def main():
    dataset = ContainerDataset()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch)
        break
if __name__ == "__main__":
    main()
