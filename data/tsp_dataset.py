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




def main():
    
    dataset = SortingDataset(size=5, num_samples=1000)
    
    all_inputs = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        input = batch['input']   #[B,L,1]
        label = batch['label']   #[B,L,1]
        all_inputs.append(input)
        all_labels.append(label)
    
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
  
    data = {
        'inputs': all_inputs,
        'labels': all_labels
    }
    
    # 保存数据
    torch.save(data, 'data/sorting_dataset.pt')
    print("数据已保存到 data/sorting_dataset.pt")

if __name__ == "__main__":
    main() 