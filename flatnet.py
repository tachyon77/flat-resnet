import torch
from torch import Tensor
import torch.nn as nn


class FlatNet (nn.Module):
    def __init__(self, input_dim, final_dim):
        self.hidden_dim = 2000
        self.linear_1 = nn.Linear(input_dim, self.hidden_dim)
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, self.final_dim)
        
        self.core_network =  nn.Sequential(
            self.linear_1,
            self.relu,
            self.bn,
            self.linear_2)

    def forward(self, x):
        out = self.core_network(x)
        out = torch.flatten(out, 1)
        return out