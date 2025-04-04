import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import pdb


class SetInterDependenceTransformer(nn.Module):
    """
    Set Interdependence Transformer module for attending between set elements
    and its permutation-invariant representation.
    """

    def __init__(self,
                 set_att_in_dim,
                 set_att_out_dim,
                 set_att_n_heads,
                 set_att_n_layers,
                 set_att_n_seeds):
        super(SetInterDependenceTransformer, self).__init__()

        # params
        self.set_att_in_dim = set_att_in_dim
        self.set_att_out_dim = set_att_out_dim
        self.set_att_n_heads = set_att_n_heads
        self.set_att_n_layers = set_att_n_layers
        self.set_att_n_seeds = set_att_n_seeds

        # layers
        self.set_attention_layers = nn.ModuleList(
            [nn.Sequential(SAB(self.set_att_out_dim,
                               self.set_att_out_dim,
                               self.set_att_n_heads, False))
             for _ in range(self.set_att_n_layers - 1)])
        self.set_pooling = PMA(self.set_att_out_dim, self.set_att_n_heads,
                               self.set_att_n_seeds)

    def forward(self, S, E):
        # concat
        Z = torch.cat([S, E], dim=1)

       
        # attend layers
        for layer in self.set_attention_layers:
            Z = layer(Z)
       
        
        # get new S
        S = self.set_pooling(Z)

        # extract new E, preventing cuda mistakes
        indices = torch.tensor([e + 1 for e in range(Z.size()[1] - 1)])
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            indices = torch.tensor([e + 1 for e in range(
                Z.size()[1] - 1)]).cuda()
        E = torch.index_select(Z, dim=1, index=indices)

        return S, E

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O



def main():
    model = SetInterDependenceTransformer(-1, 4, 2, 2, 2)
    S = torch.randn(3, 1, 4)
    E = torch.randn(3, 5, 4)
 
    output = model(S, E)
    print(output[0].size())
    print(output[1].size())

if __name__ == "__main__":
    main()


