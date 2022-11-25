"""
Pixel-Set encoder module

author: Vivien Sainte Fare Garnot
"""

import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import tanh


class DenseSetEncoder(nn.Module):
    def __init__(self, input_dim, mlp1=[10, 32, 64], len_max_seq=24):
        """
        Pixel-set encoder.
        Args:
            input_dim (int): Number of channels of the input tensors
            mlp1 (list):  Dimensions of the successive feature spaces of MLP1
        """

        super(DenseSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)

        self.name = 'DSE-{}'.format('|'.join(list(map(str, self.mlp1_dim))))

        
        print(self.name)
        assert (input_dim == mlp1[0])

        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(Dense(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)


    def forward(self, input):
        out = self.mlp1(input)
        return out

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Dense, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, input):
        out = self.lin(input)
        out = tanh(out)
        return out
