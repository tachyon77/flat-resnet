"""Train a model.

Author:
    Mohammad Mahbubuzzaman (tachyon77@gmail.com)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from dataset import ResnetOutputDataset
from flatnet import FlatNet

loss_fn = nn.MSELoss()

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using fixed seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    model = FlatNet((3,256,256), 1000)

    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    
    train_images_file='/home/mahbub/research/flat-resnet/data/train_images.pt'
    train_output_file='/home/mahbub/research/flat-resnet/data/train_output.pt'    
    train_dataset = ResnetOutputDataset(train_images_file, train_output_file)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=None)
   
    dev_images_file='/home/mahbub/research/flat-resnet/data/dev_images.pt'
    dev_output_file='/home/mahbub/research/flat-resnet/data/dev_output.pt'    
    dev_dataset = ResnetOutputDataset(dev_images_file, dev_output_file)

    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=None)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for features, y in train_loader:
                # Setup for forward                

                batch_size = args.batch_size

                optimizer.zero_grad()

                # Forward
                outputs = model(features)
                y = y.to(device)
                loss = loss_fn (outputs, y)
                                
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)                

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                                
                                
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results = evaluate(model, dev_loader, device)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                

def evaluate(model, data_loader, device):
    nll_meter = util.AverageMeter()

    model.eval()
    
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for features, y in data_loader:
            # Setup for forward            

            batch_size = y.shape[0]

            # Forward
            outputs = model(features)
            y = y.to(device)
            loss = loss_fn(outputs, y)
            nll_meter.update(loss.item(), batch_size)
            
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)
             

    model.train()

    results_list = [('NLL', nll_meter.avg)]

    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main(get_train_args())
