import torch
from torch import Tensor
import torch.nn as nn
from math import prod

class FlatNet (nn.Module):
    def __init__(self, image_shape, output_shape):
        super(FlatNet, self).__init__()
        self.input_dim = prod(image_shape)
        self.hidden_dim = 2000
        self.linear_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, output_shape)

        self.core_network =  nn.Sequential(
            self.linear_1,
            self.relu,
            self.bn,
            self.linear_2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.core_network(x)
        return out