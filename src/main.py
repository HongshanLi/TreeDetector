import argparse
import os
import random
from shutil import copyfile
import time
import warnings
import pickle
import scipy.misc
import json


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

import preprocess 
from dataset import TreeDataset, TreeDatasetInf
from models import ResNetModel, UNet
from criterion import Criterion
from train_loop import Trainer
from post_process import CleanUp
from evaluate import evaluate_model

model_names = ['resnet', 'unet']

parser = argparse.ArgumentParser(
        description='PyTorch Model For Segmenting Trees')
parser.add_argument('--root', metavar='DIR', required=True,
        help='project root directory')


parser.add_argument('--preprocess', action='store_true',
        help='preprocess raw data')

parser.add_argument('--model', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('--train', action='store_true', 
                    help='train the model')
parser.add_argument('--pretrained', action='store_true',
                    help="use pretrained resnet152")
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N',
                    help='batch size')
parser.add_argument('--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', dest='start_epoch', type=int,
                    default=1, metavar='N', 
                    help='epoch to start training')
parser.add_argument('--resume', dest='resume', metavar='PATH',
                    help='resume training from a checkpoint')
parser.add_argument('--log-dir', default='./logs', type=str, metavar="PATH",
                    help='dir to save logs')
parser.add_argument('--ckp-dir', default='./ckps', type=str, metavar="PATH",
                    help='checkpoint directory')


parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--find-best-model', dest='find_best_model', 
        action='store_true', help='find the best model from the ckp')

# evaluation
parser.add_argument('-t', '--threshold', dest='threshold',
        type=float, default=0.5, 
        help='threshold to convert softmax to one-hot encode')

# prediction
parser.add_argument('-p', '--predict', dest='predict', 
        action='store_true', help='predict mask on aerial imgs')
parser.add_argument('--model-ckp', type=str, metavar="PATH",
        help='path to the model ckp')
parser.add_argument('--images', type=str, 
        metavar="PATH",
        help='dir to store imgs to be masked')
parser.add_argument('--mask-dir', type=str, metavar="PATH",
        default='./masks',
        help='dir to save mask')



device = torch.device('cuda:0' if torch.cuda.is_available()
     else 'cpu')


def main(args, model, config):

    if not os.path.isdir(args.ckp_dir):
        os.mkdir(args.ckp_dir)

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    if not os.path.isdir(args.mask_dir):
        os.mkdir(args.mask_dir)

    if args.preprocess:
        print('dividing raw images into subimages...')
        preprocess.divide_raw_imgs_masks(config, root_dir=args.root)
        print('computing mean and standard deviation for the dataset...')
        preprocess.compute_mean_std(config, root_dir=args.root)

    # make inference on images
    if args.predict:
        predict(args=args, model=model)

    # evaluate model performance on test set
    if args.evaluate:
        evaluate(args=args, config=config, model=model)

    # train the model
    if args.train:
        train(args=args, config=config, model=model)

    # find the best model from the ckps
    if args.find_best_model:
        find_best_model(args)

def find_best_model(args):
    logpath = os.path.join(args.log_dir, 'log.pickle')
    with open(logpath, 'rb') as f:
        log = pickle.load(f)
    best_model = 1
    min_loss = log[1]['val_loss']
    for epoch, eplog in log.items():
        if eplog['val_loss'] <= min_loss:
            min_loss = eplog['val_loss']
            best_model = epoch
    print("best model is : model_{}.pth".format(best_model))
    return 

def evaluate(args, config, model):
    model = model
    transform, mask_transform = get_transform(config, proj_root=args.root)
    test_dataset = TreeDataset(config['proc_data'], 
            transform=transform, mask_transform=mask_transform,
            purpose='test')
    model.load_state_dict(
            torch.load(args.model_ckp, map_location=device))
    model = model.to(device)

    evaluate_model(test_dataset, model, 
            threshold=args.threshold, device=device, 
            batch_size=args.batch_size)
    return 

def predict(args, model):
    if args.images is None:
        raise TypeError("you need to specify image dir")
    with torch.no_grad():
        model = model
        model.load_state_dict(torch.load(
            os.path.join(args.model_ckp), 
            map_location=device,
            ))
        model = model.to(device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            # @TODO add normalization here
            ])

        ds = TreeDatasetInf(args.images, transform=transform)
        cleanup = CleanUp()
        for i in range(len(ds)):
            img, img_name = ds[i]
            img = img.unsqueeze(0)
            img = img.to(device)
            mask = model(img)
            _,_,h,w = mask.shape

            mask = mask.view(h, w)
            mask = mask.cpu().numpy()
            
            mask = cleanup(mask)

            scipy.misc.imsave(
                    os.path.join(args.mask_dir, img_name), mask
                    )
    
    return

def train(args, model, config):
    transform, mask_transform = get_transform(
            config, proj_root=args.root)
    
    # full dir for processed data
    proc_data = os.path.join(args.root, config['proc_data'])

    train_dataset = TreeDataset(proc_data,
            transform=transform, mask_transform=mask_transform,
            purpose='train')

    val_dataset = TreeDataset(proc_data, 
            transform=transform, mask_transform=mask_transform,
            purpose='val')

    
    model = model
    if args.resume is not None:
        model.load_state_dict(
                torch.load(os.path.join(
                    args.ckp_dir, args.resume)))

        # start from the end epoch of the previous 
        # training loop
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        else:
            start_epoch=len(os.listdir(args.ckp_dir))
    else:
        start_epoch=1

    criterion = Criterion()
    trainer = Trainer(train_dataset=train_dataset,
            val_dataset=val_dataset, 
            model=model, criterion=criterion, args=args)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        trainer(epoch)
        trainer.validate(epoch)
        trainer.logger.save_log()
    return 

def get_transform(config, **kwargs):
    proj_root=kwargs['proj_root']

    # get mean and std of the dataset
    mean_std_path = os.path.join(proj_root, config['mean_std'])
    with open(mean_std_path, 'r') as f:
        mean_std = json.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            tuple(mean_std['mean']),
            tuple(mean_std['std'])
            )])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    return transform, mask_transform


if __name__=="__main__":  
    args = parser.parse_args()
    config_file=os.path.join(args.root, 'config', "config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)

    if args.model=='resnet':
        model = ResNetModel(args.pretrained)
    elif args.model=='unet':
        model = UNet()

    model=model.to(device)
    main(model=model, args=args, config=config)


