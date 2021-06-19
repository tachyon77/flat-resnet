"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Mohammad Mahbubuzzaman (tachyon77@gmail.com)s
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

import config
from collections import OrderedDict
from json import dumps
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from dataset import MyDataset
import resnet

def get_model_output(model, dataset):
    
    trained_resnet = model
    trained_resnet = trained_resnet.to(device)
    trained_resnet.eval()

    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    with torch.no_grad(), tqdm(total=len(dataset)) as progress_bar:
         for features, y, in data_loader:            
            output = trained_resnet(features)           
            progress_bar.update(batch_size)
            yield output


# This will be used to generate training dataset for flatnet.

def get_resnet_output(trained_resnet, dataset, output_path):
    # preallocate a tensor of N, W,H,C dimensions for N Images 
    N = len(dataset)
    D = 100 # TODO
    output = torch.Tensor(N, D)

    # iterate over output from model and put them in output tensor
    start = 0
    for batch_output in get_model_output(trained_resnet, dataset):
        output[start:start+batch_size, :] = batch_output
        start = start + batch_size

    torch.save(output, output_path)    



class RandomDataset(Dataset):
    def __init__(self, range):
        self.data = torch.rand(dims)*range

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def build_train_dataset_using_resnet():
    model = resnet.resnet50()

    model = nn.DataParallel(model, gpu_ids)
    
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = MyDataset(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)


def main():

if __name__ == '__main__':
    main(get_inference_args())
    