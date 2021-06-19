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
    def __init__(self, data_path):
        super(ImageDataset, self).__init__()

        self.dataset = torch.load(data_path)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.dataset.shape[0] # Since first dimension is the N


class ResnetOutputDataset(data.Dataset):
    """Random Dataset.

    The dataset is a tensor with first dimension as N and remaining as image dimension.

    Args:
        data_path (str): Path to .npz file containing dataset.
    """
    def __init__(
        self, 
        input_data_path='/home/mahbub/research/flat-resnet/random_images.npz',
        output_data_path='/home/mahbub/research/flat-resnet/10000_random_chw_tensors.npz'):
        super(ResnetOutputDataset, self).__init__()

        self.input_tensors = torch.load(input_data_path)
        self.output_tensors = torch.load(output_data_path)

    def __getitem__(self, idx):
        return (self.input_tensors[idx], self.output_tensors[idx])

    def __len__(self):
        return self.input_tensors.shape[0] # Since first dimension is the N




