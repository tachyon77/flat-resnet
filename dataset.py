"""Dataset class.

Author:
    Mohammad Mahbubuzzaman (tachyon77@gmail.com)
"""
import logging
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import dataset
import torch
import ujson as json
from collections import Counter


class ImageDataset(data.Dataset):
    """Random Dataset.

    The dataset is a tensor with first dimension as N and remaining as image dimension.

    Args:
        data_path (str): Path to .npz file containing dataset.
    """
    def __init__(self, data_path='/home/mahbub/research/flat-resnet/random_images.npz'):
        super(ImageDataset, self).__init__()

        self.dataset = torch.load(data_path)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.dataset.shape[0] # Since first dimension is the N



