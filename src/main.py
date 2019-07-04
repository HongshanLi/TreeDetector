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

parser.add_argument('--preprocess', action='store_true',
        help='preprocess raw data')

parser.add_argument('--model', metavar='model', default='resnet',
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

parser.add_argument('--resume', action='store_true',
                    help='resume training from a checkpoint')


parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--find-best-model', dest='find_best_model', 
        action='store_true', help='find the best model from the ckp')

# evaluation
parser.add_argument('--baseline', action='store_true',
        help="evaluate model with a baseline")

parser.add_argument('-t', '--threshold',
        type=float, default=0.5, 
        help='threshold to convert softmax to one-hot encode')

# prediction
parser.add_argument('-p', '--predict', dest='predict', 
        action='store_true', help='predict mask on aerial imgs')
parser.add_argument('--model-ckp', type=str, metavar="PATH",
        help='path to the model ckp')
parser.add_argument('--image-dir', type=str, 
        metavar="PATH",
        help='dir to store imgs to be masked')
parser.add_argument('--mask-dir', type=str, metavar="PATH",
        default='./masks',
        help='dir to save mask')


args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available()
     else 'cpu')

# get project root directory
ROOT=os.path.join("../", os.getcwd())

if args.model == 'resnet':
    model = ResNetModel(args.pretrained)
    CKPDIR = os.path.join(ROOT, 'resnet_ckps/')
    LOGDIR = os.path.join(ROOT, 'resnet_logs/')
elif args.model == 'unet':
    model = UNet()
    CKPDIR = os.path.join(ROOT, 'unet_ckps/')
    LOGDIR = os.path.join(ROOT, 'unet_logs/')
else:
    raise("model must either be resnet or unet")

model.to(device)

# config
config_file=os.path.join(ROOT, 'config', "config.json")
with open(config_file, 'r') as f:
    config = json.load(f)

def main():
    if not os.path.isdir(CKPDIR):
        os.mkdir(CKPDIR)

    if not os.path.isdir(LOGDIR):
        os.mkdir(LOGDIR)
    
    if not os.path.isdir(args.mask_dir):
        os.mkdir(args.mask_dir)

    # preprocess data
    if args.preprocess:
        preprocess()
        

    # make inference on images
    if args.predict:
        predict()
        return

    # evaluate model performance on test set
    if args.evaluate:
        evaluate()
        return

    # train the model
    if args.train:
        train()
        return

    # find the best model from the ckps
    if args.find_best_model:
        find_best_model()
        return

    return

def preprocess():
    print('dividing raw images into subimages...')
    preprocess.divide_raw_imgs_masks(config, root_dir=ROOT)
    print('computing mean and standard deviation for the dataset...')
    preprocess.compute_mean_std(config, root_dir=ROOT)
    return

def find_best_model():
    logpath = os.path.join(LOGDIR, 'log.pickle')
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

def evaluate():
    transform, mask_transform = get_transform()
    test_dataset = TreeDataset(config['proc_data'], 
            transform=transform, mask_transform=mask_transform,
            purpose='test')

    # baseline
    if args.baseline:
        print("====== Performance of baseline model ======")
        baseline = PixelThreshold(config['greenpixel'])
        evalute_model(test_dataset, baseline, 
                threshold=args.threshold,
                batch_size=args.batch_size,
                device=torch.device('cpu'))
    
    # CNN model

    model.load_state_dict(
            torch.load(args.model_ckp, map_location=device))
    print("====== Performance of CNN model ======")
    evaluate_model(test_dataset, model, 
            threshold=args.threshold, device=device, 
            batch_size=args.batch_size)
    return 

def predict():
    if args.image_dir is None:
        raise TypeError("you need to specify directory of images")
    with torch.no_grad():
        model.load_state_dict(torch.load(
            os.path.join(args.model_ckp), 
            map_location=device,
            ))

        transform = transforms.Compose([
            transforms.ToTensor(),
            ])

        ds = TreeDatasetInf(args.image_dir, transform=transform)
        cleanup = CleanUp()
        for i in range(len(ds)):
            img, img_name = ds[i]

            # normalize img
            # @TODO implement std by hand to avoid calculate 
            # mean twice 
            mean = torch.mean(img, dim=(1,2))
            std = torch.std(img, dim=(1,2))

            img = (img - mean) / std

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

def train():
    transform, mask_transform = get_transform()
    
    # full dir for processed data
    proc_data = os.path.join(ROOT, config['proc_data'])

    train_dataset = TreeDataset(proc_data,
            transform=transform, mask_transform=mask_transform,
            purpose='train')

    val_dataset = TreeDataset(proc_data, 
            transform=transform, mask_transform=mask_transform,
            purpose='val')

    num_models = len(os.listdir(CKPDIR))
    if args.resume:
        lastest_model = 'model_{}.pth'.format(num_models)

        ckp_path = os.path.join(CKPDIR, lastest_model)
        model.load_state_dict(
                torch.load(ckp_path, map_location=device))
        start_epoch = num_models
    else:
        # start training from epoch 1 
        # remove all existing ckps
        start_epoch=1
        if num_models > 1:
            print("Removing existing ckps in {}, this may take a while.".format(CKPDIR))
            for ckp in os.listdir(CKPDIR):
                os.remove(os.path.join(CKPDIR, ckp))


    criterion = Criterion()
    trainer = Trainer(train_dataset=train_dataset,
            val_dataset=val_dataset, 
            model=model, criterion=criterion,
            ckp_dir = CKPDIR,
            log_dir = LOGDIR,
            batch_size=args.batch_size,
            lr=args.lr,
            threshold=args.threshold,
            start_epoch=start_epoch,
            resume=args.resume,
            epochs=args.epochs,
            print_freq=args.print_freq)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        trainer(epoch)
        trainer.validate(epoch)
        trainer.logger.save_log()
    return 

def get_transform():
    # get mean and std of the dataset
    mean_std_path = os.path.join(ROOT, config['mean_std'])
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
    main()


