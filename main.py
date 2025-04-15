from model.encoder import Encoder
from model.sortTransfoemer import SortingTransformer
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pdb

import torch
import torch.nn as nn

# 定义参数
seq_length = 5  # 序列长度
input_dim = 10  # 输入特征维度
num_heads = 2  # 注意力头的数量
batch_size = 3  # 批次大小

# 创建输入序列
query = torch.randn(1, batch_size, input_dim)
key = torch.randn(seq_length, batch_size, input_dim)
value = torch.randn(seq_length, batch_size, input_dim)

# 创建多头注意力模块
multihead_attn = nn.MultiheadAttention(input_dim, num_heads)

# 进行多头注意力计算
attn_output, attn_output_weights = multihead_attn(query, key, value)

pdb.set_trace()
# 验证每行的和是否接近 1
row_sums = attn_output_weights.sum(dim=-1)
print("每行的和:", row_sums)
print("是否每行的和都接近 1:", torch.allclose(row_sums, torch.ones_like(row_sums)))
    
