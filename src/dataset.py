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


def _create_img_ids(img_dir):
    '''create ids for each raw HD imgs'''
    img_ids = []
    for x in os.listdir(img_dir):
        img_ids = img_ids + os.listdir(
                os.path.join(img_dir, x))
    return img_ids

def preprocess_imgs(img_dir, out_dir):
    '''divide raw HD images (1250x1250) into subimgs of dim 250 x 250
    Only keep rgb channels
    Args:
        img_dir: root dir of raw images 
        out_dir: root dir for preprocessed imgs and masks

    Return: of subimgs and submasks in the following structure
    
    Trees_processed 
        imgs
        masks
    '''

    if not os.path.isdir(out_dir):
        try:
            os.makedirs(os.path.join(out_dir, 'imgs/'))
            os.makedirs(os.path.join(out_dir, 'masks/'))
        except OSError as e:
            print(e)

    img_ids = _create_img_ids(img_dir)
    
    num = 1250 // 250
    for img_id in img_ids:
        # img sub-folder
        img_sfd = img_id.split('-')[0]

        # rgb
        img_rgb = img_id + "_RGB-Ir.tif"
        
        # mask
        img_mask = img_id + "_TREE.png"
        
        img_path = os.path.join(img_dir, 
                img_sfd, img_id, img_rgb)


        mask_path = os.path.join(img_dir,
                img_sfd, img_id, img_mask)

        img = io.imread(img_path)
        mask = io.imread(mask_path)
        
        # divid into subimgs and save 
        for i in range(num):
            for j in range(num):
                start_x, end_x = i*250, (i+1)*250
                start_y, end_y = j*250, (j+1)*250
                
                sub_img = img[start_x:end_x, start_y:end_y,0:3]

                sub_mask = mask[start_x:end_x, start_y:end_y,0:3]
                
                idx = '{}_{}'.format(i,j)
                file_name = img_id + '_' + idx + ".png"

                # save img
                Image.fromarray(sub_img).save(
                        os.path.join(out_dir, 'imgs/', file_name)
                        )

                # save mask
                Image.fromarray(sub_mask).save(
                        os.path.join(out_dir, 'masks/', file_name)
                        )
    print("Num of imgs processed:", len(img_ids))


def compute_mean_std(dataset):
    '''compute channelwise mean and std of the dataset
    Arg:
        dataset: pytorch Dataset where each data is a tensorized PIL img
    '''
    batch_size=5
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    start = time.time()

    # compute means for each channels
    mean = 0

    for i,(img, elv, mask) in enumerate(loader):
        mean = mean + torch.mean(img, dim=[0,2,3])
    
    mean = mean /(i+1)
    
    # compute std
    cum_var = 0
    _mean = mean.view(1, 3, 1, 1).repeat(batch_size, 1, 250,250) 
    for i,(img, elv, mask) in enumerate(loader):
        # repeat means across channels
        
        try:
            cum_var = cum_var + torch.mean((img - _mean)**2, dim=[0,2,3])
        except:
            print('current img:', img.shape)
            print('current mean', _mean.shape)
    var = cum_var / (i + 1)

    std = torch.sqrt(var)
    
    return mean, std


    
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4137, 0.4233, 0.3968),
        (0.2275, 0.2245, 0.2424))
    ])

elv_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])

class TreeDataset(Dataset):
    def __init__(self, proc_dir, transform=transform, 
        mask_transform=mask_transform, purpose='train'):
        
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
            mask = mask[1,:,:].view(1,250,250)

        return img, mask
    
    def get_file_names(self):
        file_names = os.listdir(os.path.join(
            self.proc_dir, 'imgs'))
        return file_names

class TreeDatasetInf(Dataset):
    '''Data fetch pipeline for inference'''
    def __init__(self, img_dir, transform=transform):
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
