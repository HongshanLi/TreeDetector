import argparse
import os
import random
from shutil import copyfile
import time
import warnings
import pickle
import scipy.misc
import json
import shutil

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

import preprocess as prep
from dataset import TreeDataset, TreeDatasetInf
from models import ResNetModel, UNet
from criterion import Criterion
from train_loop import Trainer
from post_process import CleanUp
from evaluate import evaluate_model
from baseline import PixelThreshold

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

parser.add_argument('--debug', action='store_true',
        help='use this flag to debug, no ckp or log will be saved')

parser.add_argument('--train', action='store_true', 
                    help='train the model')
parser.add_argument('--pretrained', action='store_true',
                    help="use pretrained resnet152")
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='N',
                    help='batch size')
parser.add_argument('--learning-rate', default=0.0001, type=float,
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


# evaluate
parser.add_argument('--use-lidar', action='store_true',
        help='use lidar image for post process')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--baseline', action='store_true',
        help='use baseline to evaluate model')
parser.add_argument('--find-best-model', dest='find_best_model', 
        action='store_true', help='find the best model from the ckp')

parser.add_argument('-t', '--threshold', dest='threshold',
        type=float, default=0.5, 
        help='threshold to convert softmax to one-hot encode')

# prediction
parser.add_argument('-p', '--predict', dest='predict', 
        action='store_true', help='predict mask on aerial imgs')
parser.add_argument('--model-ckp', type=str, metavar="PATH",
        help='path to the model ckp')
parser.add_argument('--image-dir', type=str, 
        metavar="PATH",
        help='dir of rgb images')
parser.add_argument('--lidar-dir', type=str,
        metavar="PATH",
        help='dir of lidar images')
parser.add_argument('--mask-dir', type=str, metavar="PATH",
        default='./masks',
        help='dir to save mask')


args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available()
     else 'cpu')

# get project root directory
ROOT=os.path.join("../", os.getcwd())

print("Use Lidar: ", args.use_lidar)
if args.model == 'resnet':
    model = ResNetModel(
            pretrained=args.pretrained,
            use_lidar=args.use_lidar)
    if args.use_lidar:
        ckp_dir = 'resnet_lidar_ckps/'
        log_dir = 'resnet_lidar_logs/'
    else:
        ckp_dir = 'resnet_ckps/'
        log_dir = 'resnet_logs/'
elif args.model == 'unet':
    if args.use_lidar:
        ckp_dir = 'unet_lidar_ckps/'
        log_dir = 'unet_lidar_logs/'
    else:
        ckp_dir = 'unet_ckps/'
        log_dir = 'unet_logs/'

    model = UNet(use_lidar=args.use_lidar)    
else:
    raise("model must either be resnet or unet")



CKPDIR = os.path.join(ROOT, ckp_dir)
LOGDIR = os.path.join(ROOT, log_dir)

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
    # directory of processed data 
    proc_dir = os.path.join(ROOT, config['proc_data'])

    if os.path.isdir(proc_dir):
        print('proc_data/ already exists in root dir, do you want to overwrite it?')
        x = input("Type y to overwrite, type any other string to exit: ")
        if x == 'y':
            shutil.rmtree(proc_dir)
            os.makedirs(os.path.join(ROOT, config['proc_imgs']))
            os.makedirs(os.path.join(ROOT, config['proc_lidars']))
            os.makedirs(os.path.join(ROOT, config['proc_masks']))
        else:
            return
    else:
        os.makedirs(os.path.join(ROOT, config['proc_imgs']))
        os.makedirs(os.path.join(ROOT, config['proc_lidars']))
        os.makedirs(os.path.join(ROOT, config['proc_masks']))

    print('====== dividing raw images into subimages ======')
    prep.divide_raw_data(config, root_dir=ROOT)
    print('====== Computing mean and standard deviation for the RGB images ======')
    prep.compute_mean_std(config, root_dir=ROOT, data_type='rgb')
    print('====== Computing mean and standard deviation for the LiDAR images ======')
    prep.compute_mean_std(config, root_dir=ROOT, data_type='lidar')
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
    transform, lidar_transform, mask_transform = get_transform()
    test_dataset = TreeDataset(config['proc_data'], 
            transform=transform, 
            lidar_transform=lidar_transform,
            use_lidar=args.use_lidar,
            mask_transform=mask_transform,
            purpose='test')

    # baseline
    if args.baseline:
        print('====== Baseline Performance ======')
        baseline = PixelThreshold(config['greenpixel'])
        evaluate_model(test_dataset, baseline,
                threshold=args.threshold, 
                device=torch.device('cpu'),
                batch_size=args.batch_size)

    print('====== CNN Model Performance ======')
    model.load_state_dict(
            torch.load(args.model_ckp, map_location=device))

    evaluate_model(test_dataset, model, 
            threshold=args.threshold, 
            use_lidar=args.use_lidar,
            device=device, 
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

        ds = TreeDatasetInf(
                img_dir=args.image_dir,
                lidar_dir=args.lidar_dir,
                transform=transform, 
                lidar_transform=transform,
                use_lidar=args.use_lidar)

        cleanup = CleanUp(threshold=args.threshold)
        for i in range(len(ds)):
            img_name, img, lidar = ds[i]

            mean = torch.mean(img, dim=(1,2))
            std = torch.std(img, dim=(1,2))
           
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
            img = (img - mean) / std

            img = img.unsqueeze(0)
            img = img.to(device)
            mask = model(img)

            if args.use_lidar:
                lidar = lidar.to(device)
                mask = mask*(1 - lidar)


            _,_,h,w = mask.shape

            mask = mask.view(h, w)
            mask = mask.cpu().numpy()
            
            mask = cleanup(mask)
            

            mask.save(
                    os.path.join(args.mask_dir, img_name)
                    )
    
    return

def train():
    transform, lidar_transform, mask_transform = get_transform()
    
    # full dir for processed data
    proc_data = os.path.join(ROOT, config['proc_data'])

    train_dataset = TreeDataset(proc_data,
            transform=transform, 
            lidar_transform=lidar_transform,
            mask_transform=mask_transform,
            use_lidar=args.use_lidar,
            purpose='train')

    val_dataset = TreeDataset(proc_data, 
            transform=transform, 
            lidar_transform=lidar_transform,
            mask_transform=mask_transform,
            use_lidar=args.use_lidar,
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

        if num_models > 1 and args.debug==False:
            print("Removing existing ckps in {}, this may take a while.".format(CKPDIR))
            for ckp in os.listdir(CKPDIR):
                os.remove(os.path.join(CKPDIR, ckp))

    criterion = Criterion()
    trainer = Trainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset, 
            model=model, 
            criterion=criterion,
            ckp_dir = CKPDIR,
            log_dir = LOGDIR,
            debug=args.debug,
            use_lidar=args.use_lidar,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            start_epoch=start_epoch,
            resume=args.resume,
            epochs=args.epochs,
            print_freq=args.print_freq)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        start = time.time()
        trainer(epoch)
        trainer.validate(epoch)
        end = time.time()

        print("Time to train one epoch is: {:0.2f}".format(end - start))
        if args.debug==False:
            trainer.logger.save_log()

    return 

def get_transform():
    # get mean and std of the dataset
    mean_std_rgb = os.path.join(ROOT, config['mean_std_rgb'])
    with open(mean_std_rgb, 'r') as f:
        mean_std = json.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            tuple(mean_std['mean']),
            tuple(mean_std['std'])
            )])


    mean_std_lidar = os.path.join(ROOT, config['mean_std_lidar'])
    with open(mean_std_lidar, 'r') as f:
        mean_std = json.load(f)

    lidar_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    return transform, lidar_transform, mask_transform


if __name__=="__main__":  
    main()


