import imageio
import scipy
import PIL.Image as Image
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as ds
import torchvision.transforms as transforms

import numpy as np
import PIL.Image as Image
import os
import skimage.io as io
import random


class TreeDataset(Dataset):
    def __init__(self, proc_dir, transform, mask_transform, 
            purpose='train'):
        '''
        Args:
            proc_dir: directory for processed data
            transform: transforms applied to imgs
            mask_transform: transforms applied to masks
        '''
        
        # get a list of img files
        self.proc_dir = proc_dir
        self.file_names = self.get_file_names()
        
        self.transform = transform
        self.mask_transform = mask_transform

        # choose the same 90% imgs for training
        np.random.seed(42)
        total_size = len(self.file_names)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = int(total_size * 0.1)

        dist = []
        for i in range(total_size):
            if i < train_size:
                dist.append(0)
            elif i>= 1 and i <train_size + val_size:
                dist.append(1)
            elif i>= train_size + val_size and i < total_size:
                dist.append(2)

        train_set = []
        val_set = []
        test_set = []
        for i, b in enumerate(dist):
            if b == 0:
                train_set.append(self.file_names[i])
            elif b==1:
                val_set.append(self.file_names[i])
            elif b==2:
                test_set.append(self.file_names[i])
        
        
        if purpose=='train':
            self.file_names = train_set
        elif purpose=='val':
            self.file_names = val_set
        elif purpose=='test':
            self.file_names = test_set
        elif purpose==None:
            pass
        else:
            print("purpose must be 'train', 'val' or 'test'")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # img subfolder
        file_name = self.file_names[idx]

        img = os.path.join(self.proc_dir, 'imgs/', file_name)
        mask = os.path.join(self.proc_dir, 'masks/', file_name)

        img = io.imread(img)
        mask = io.imread(mask)

        if self.transform is not None:
            img  = self.transform(img)
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask
    
    def get_file_names(self):
        file_names = os.listdir(os.path.join(
            self.proc_dir, 'imgs'))
        return file_names

class TreeDatasetInf(Dataset):
    '''Data fetch pipeline for inference'''
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = io.imread(os.path.join(self.img_dir, img_name))
        
        if self.transform is not None:
            img = self.transform(img)

        return img, img_name

        


if __name__=='__main__':
    def test_dataset():
        img_dir = '/home/hongshan/data/Trees'
        out_dir = '/home/hongshan/data/Trees_processed'
        
        d = TreeDataset(out_dir)


        for i in range(10):
            x,y,z,fn = d[i]
            st.write("Image {}".format(fn))
            s = z[:,:,1]
            img = Image.fromarray(x)
            elv = Image.fromarray(y)
            full = Image.fromarray(z)
            single = Image.fromarray(s)
            st.image([img, elv, full, single])
        return

    def test_compute_mean_std(dataset):
        compute_mean_std(dataset)
        return

    img_dir='/mnt/efs/Trees'
    proc_dir = '/mnt/efs/Trees_processed/'
    preprocess_imgs(img_dir, proc_dir)
