import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import pdb

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, num_heads,dim_feedforward = 512, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self,inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

class PointerTransformerDecoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super(PointerTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, num_heads,dim_feedforward = 512, batch_first=True) 
            for _ in range(num_layers)
        ])

        self.attn_pointer = nn.Linear(d_model * 2, 1)  # 指针注意力计算

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):

        for layer in self.layers:
            tgt = layer(tgt,memory,tgt_mask,memory_mask)

        scores = self.attn_pointer(torch.cat([tgt, memory], dim=-1)).squeeze(-1)
        return scores




class SetTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super(SetTransformer, self).__init__()
        self.encoder = TransformerEncoder(dim, num_heads, num_layers)
    
    def forward(self,inputs):
      
        output = self.encoder(inputs)
        pdb.set_trace()


        return output



def main():
    model = SetTransformer(8,2,2)
    inputs = torch.randn(3,2,8)
    output = model(inputs)
    print(output.size())


if __name__ == "__main__":
    main()


