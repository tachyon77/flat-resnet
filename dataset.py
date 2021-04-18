"""Dataset class.

Author:
    Mohammad Mahbubuzzaman (tachyon77@gmail.com)
"""
import logging
import os
import queue
import re
import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import ujson as json
from collections import Counter


class MyDataset(data.Dataset):
    """My Dataset.

    Each item in the dataset is a tuple with the following entries (in order):        
        - y: Correct label.
        - id: ID of the example.

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
    """
    def __init__(self, data_path):
        super(MyDataset, self).__init__()

        dataset = np.load(data_path)
       
        self.ys = torch.from_numpy(dataset['ys']).long()    
        self.ids = torch.from_numpy(dataset['ids']).long()

    def __getitem__(self, idx):
        idx = self.ids[idx]
        example = (            
            self.ys[idx],            
            self.ids[idx]
        )

        return example

    def __len__(self):
        return len(self.ids)

