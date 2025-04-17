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
import numpy as np



for i in range(10):
    torch.manual_seed(42)
    shuffle_idx = torch.randperm(5)
    print(shuffle_idx)
        
