"""Test a model.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Mohammad Mahbubuzzaman (tachyon77@gmail.com)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
import resnet

from args import get_test_args
from dataset import ImageDataset
from collections import OrderedDict
from json import dumps
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load


def main(args):

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Get model
    #log.info(f'Loading checkpoint from {args.load_path}...')

    model = resnet.resnet50()
    model = nn.DataParallel(model, gpu_ids)
    #log.info(f'Loading checkpoint from {args.load_path}...')
    #model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('loading dataset...')
    input_data_file = '/home/mahbub/research/flat-resnet/10000_random_chw_tensors.npz' 
                        #vars(args)[f'{args.input_data_file}']
    dataset = ImageDataset(input_data_file)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=None)

    #class_label_file = '/home/mahbub/research/flat-resnet/imagenet_classes.txt'
    # Read the categories
    #with open(class_label_file, "r") as f:
    #    categories = [s.strip() for s in f.readlines()]

    # Evaluate
    log.info(f'Running inference ...')

    output = torch.zeros(len(dataset), 1000) # TODO: 1000 is number of class or resnet output size, remove hard coding.
    out_idx = 0
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
         for images in data_loader:
            # Setup for forward
            images = images.to(device)
  
            batch_size = images.shape[0]
            #print ("batch size is {}".format(batch_size))

            #print("Input is : {}".format(images[0,0,0,:10]))
            # Forward
            output[out_idx:out_idx+batch_size] = model(images)
            out_idx += batch_size
            
            #print("output shape is {}".format(output.shape))
            #print("Output is: {}".format(output))

            #probabilities = torch.nn.functional.softmax(output, dim=1)
            #print("probabilities shape is {}".format(probabilities.shape))
            #print ("probabilities sum = {}".format(probabilities.sum(axis=1)))

            # Show top categories per image
            #K = 5
            #top_prob, top_catid = torch.topk(probabilities, K)

            #print("top catid shape is {}".format(top_catid.shape))

            #for i in range(top_prob.shape[0]):
            #    for k in range(K):
            #        print(categories[top_catid[i,k]], top_prob[i,k].item())

            # Log info
            progress_bar.update(batch_size)

    # Write output to a file
    torch.save(output, "resnet50_output")
if __name__ == '__main__':
    main(get_test_args())
