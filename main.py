import argparse
import os
import random
import shutil
import time
import warnings
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import DataLoader

import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dataset import TreeDataset
from model import Model
from criterion import Criterion
from train_loop import Trainer


"""Arguments:
    num_workers
    batch_size
    start_epoch: which epoch to start
    epochs: num of epochs to train
    lr: learning rate
    log_dir: directory to write log files
    ckp_dir: directory to save checkpoints
    print_step: num of step to print progess

"""
# automatically add ckp_dir to .gitignore
# automatically add log_dir to .gitignore 

#TODO give a list of possible pretrained models
model_names = []

parser = argparse.ArgumentParser(description='PyTorch Model For Segmenting Trees')
parser.add_argument('--data', metavar='DIR', required=False,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', dest='resume', default='./ckps/best_model.pth',
                    metavar='PATH',
                    help='resume training from a checkpoint')
parser.add_argument('--log-dir', default='./logs', type=str, metavar="PATH",
                    help='dir to save logs')
parser.add_argument('--ckp-dir', default='./ckps', type=str, metavar="PATH",
                    help='checkpoint directory')

parser.add_argument('--train', dest='train', action='store_true', 
                    help='train the model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--find-best-model', dest='find_best_model', 
        action='store_true', help='find the best model from the ckp')


parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# evaluation
parser.add_argument('-t', '--threshold', dest='threshold',
        type=float, default=0.5, 
        help='threshold to convert softmax to one-hot encode')

def main():
    args = parser.parse_args()

    if not os.path.isdir(args.ckp_dir):
        os.mkdir(args.ckp_dir)

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    
    '''
    validator = Validator(val_dataset=val_dataset, model=model, 
            criterion=criterion, args=args)
    '''

    if args.evaluate:
        # print some stuff to show model size
        # GPU memories need to hold the model
        # amount of data 
        # print total training steps for one epoch

        return


    if args.train:
        train_dataset = TreeDataset(args.data, purpose='train')
        val_dataset = TreeDataset(args.data, purpose='val')
        model = Model()
        criterion = Criterion()
        trainer = Trainer(train_dataset=train_dataset,
                val_dataset=val_dataset, 
                model=model, criterion=criterion, args=args)

        for epoch in range(args.start_epoch, args.epochs+1):
            trainer(epoch)
            trainer.validate(epoch)
            trainer.logger.save_log()

    if args.find_best_model:
        # find the best models from checkpoint
        logpath = os.path.join(args.log_dir, 'log.pickle')
        with open(logpath, 'rb') as f:
            log = pickle.load(f)

        print(log)
        
        best_model = 1
        min_loss = log[1]['loss']
        for k, v in log.items():
            if v['loss'] <= min_loss:
                min_loss = v['loss']
                best_model = k
        print("best model is : model_{}.pth".format(best_model))
        
        bpth=os.path.join(args.ckp_dir, 'best_model.pth')
        print("It has been copied to {}".format(bpth))

            




if __name__=="__main__":
    main()


