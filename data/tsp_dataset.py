import torch
import numpy as np
import pdb
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import math

class SortingDataset(Dataset):
    def __init__(self, size=20, num_samples=1000,seed=1234):
        self.size = size
        self.dim = math.ceil(math.log2(self.size))
        self.num_samples = num_samples

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(seed)
        # 生成不重复的随机数序列
        self.data = torch.zeros((self.num_samples, self.size, 1, self.dim), device=device)  # 增加32位二进制维度
        for i in range(self.num_samples):
            random_seq = torch.randperm(self.size)[:self.size]
            # 将每个数字转换为32位二进制表示
            binary_seq = torch.zeros((self.size, self.dim), device=device)
            for j in range(self.size):
                binary_seq[j] = torch.tensor([int(b) for b in format(random_seq[j].item(), '0'+str(self.dim)+'b')], device=device)
            self.data[i] = binary_seq.view(-1, 1, self.dim)
        
        self.data = self.data.squeeze()
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        weights = 2 ** torch.arange(self.dim - 1, -1, -1).to(self.data.device)
        values = (self.data[idx] * weights).sum(dim=-1)
        # 获取排序后的索引作为标签
        label = torch.argsort(values, dim=0)

        data = dict(
            input = self.data[idx],
            label = label
        )

        return data



class SortDataset2(Dataset):
    def __init__(self, num_samples, sequence_length):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机输入序列
        input_seq = torch.rand(self.sequence_length, 1) # Shape: [L, 1]
        # 生成排序后的索引标签
        # argsort 返回的是从小到大排序的原始索引
        label_seq = torch.argsort(input_seq.squeeze(-1)) # Shape: [L]

        input_seq = input_seq.to(self.device)
        label_seq = label_seq.to(self.device)

        return {'input': input_seq, 'label': label_seq}

def main():
    
    train_dataset = SortDataset2(num_samples=10000, sequence_length=15)
    test_dataset = SortDataset2(num_samples=1000, sequence_length=15)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch in train_dataloader:
        input = batch['input']
        label = batch['label']
        print(input.shape, label.shape)
        break

if __name__ == "__main__":
    main() 