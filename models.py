"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class FlatResnet(nn.Module):
    """Resnet In a Layer.

    Args:        
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, hidden_size, drop_prob=0.):
        super(FlatResnet, self).__init__()
    

        hidden_size = 3 * hidden_size
        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     is_bidirectional = True,
                                     drop_prob=drop_prob)

            

    def forward(self, features):
       
        return None
