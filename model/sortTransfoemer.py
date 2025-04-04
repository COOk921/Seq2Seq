import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define Transformer Model for Sorting
class SortingTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(SortingTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), num_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, output_dim)  # Predict sorted indices
        self.embedding = nn.Linear(input_dim, d_model)
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, input, mask):
        x = self.embedding(input) 
        x = self.encoder(x)  # [B,L,L]

        logits = self.decoder(x)    
        logits = logits.permute(0, 2, 1)
        logits.masked_fill(mask, -float('inf'))
        return logits