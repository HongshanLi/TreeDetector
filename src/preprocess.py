import csv
import os
import shutil
import argparse
import json
import skimage.io as io
import PIL.Image as Image
from dataset import TreeDataset
import numpy as np

from torch.utils.data import DataLoader


def divide_raw_imgs_masks(config, **kwargs):
    '''divide raw HD images (1250x1250) into subimgs of dim 250 x 250
    Only keep rgb channels
    
    '''
    root_dir = kwargs['root_dir']
    # directory of processed data 
    proc_dir = os.path.join(root_dir, config['proc_data'])

    if os.path.isdir(proc_dir):
        print('proc_data/ already exists in root dir, do you want to overwrite it?')
        x = input("Type y to overwrite, type any other string to exit: ")
        if x == 'y':
            shutil.rmtree(proc_dir)
            os.makedirs(os.path.join(root_dir, config['proc_imgs']))
            os.makedirs(os.path.join(root_dir, config['proc_masks']))
        else:
            return
    else:
        os.makedirs(os.path.join(root_dir, config['proc_imgs']))
        os.makedirs(os.path.join(root_dir, config['proc_masks']))

    # num of subimages to divide horizontally
    num_hor = config['raw_img_width'] // config['proc_img_width']

    # num of subimages to divide vertically
    num_ver = config['raw_img_height'] // config['proc_img_height']

    ph = config['proc_img_height']
    pw = config['proc_img_width']
    with open(os.path.join(root_dir, config['raw_img_mask']), "r") as f:
        data_paths = csv.reader(f, delimiter=',')
        for i, path in enumerate(data_paths):
            # process one image
            img_path, mask_path = path[0], path[1]
            img, mask=io.imread(img_path), io.imread(mask_path)

            for j in range(num_hor):
                for k in range(num_ver):
                    start_x, end_x = j*pw, (j+1)*pw
                    start_y, end_y = k*ph, (k+1)*ph
                    
                    sub_img = img[start_x:end_x, start_y:end_y,0:3]
                    sub_mask = mask[start_x:end_x, start_y:end_y,1]
                    
                    file_name = "{}_{}_{}.png".format(i, j, k)

                    # save img
                    Image.fromarray(sub_img).save(
                        os.path.join(root_dir, config['proc_imgs'],file_name)
                        )

                    # save mask
                    Image.fromarray(sub_mask).save(
                        os.path.join(root_dir, config['proc_masks'],file_name)
                        )
    return 
                
def compute_mean_std(config, **kwargs):
    '''compute channelwise mean and std of the dataset
    '''
    proj_root=kwargs['root_dir']
    imgs = os.listdir(
            os.path.join(proj_root, config['proc_imgs'])
            )


    img_paths = [
            os.path.join(proj_root, config['proc_imgs'],img) 
            for img in imgs]
    
    # compute mean

    # sum of pixel value throughout the entire data set
    total = np.array([0, 0, 0])
    for imp in img_paths:
        img = io.imread(imp).astype(np.float32)/255
        _mean = np.mean(img, axis=(0,1))
        total = total + _mean

    mean = total / len(imgs)

    # compute standard deviation
    
    # Total variance
    total_var = np.array([0, 0, 0])
    for imp in img_paths:
        img = io.imread(imp).astype(np.float32)/255
        _var = np.mean((img - mean)**2, axis=(0,1))
        total_var = total_var + _var
    
    var = total_var / len(imgs)
    std = np.sqrt(var)

    # save mean and std to json
    mean_std = {
        "mean": np.ndarray.tolist(mean),
        "std": np.ndarray.tolist(std)
        }

    with open(os.path.join(proj_root, config['mean_std']), 'w') as f:
        json.dump(mean_std, f)

    return
        
            
