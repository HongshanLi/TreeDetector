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


def divide_raw_data(config, **kwargs):
    '''divide raw HD images (1250x1250) into subimgs of dim 250 x 250
    Only keep rgb channels
    
    '''
    root_dir = kwargs['root_dir']
    # num of subimages to divide horizontally
    num_hor = config['raw_img_width'] // config['proc_img_width']

    # num of subimages to divide vertically
    num_ver = config['raw_img_height'] // config['proc_img_height']

    ph = config['proc_img_height']
    pw = config['proc_img_width']
    with open(os.path.join(root_dir, config['raw_data_path']), "r") as f:
        data_paths = csv.reader(f, delimiter=',')
        for i, data in enumerate(data_paths):
            # process one image
            #img_path, lidar_path, mask_path = path[0], path[1], path[2]
            #img, lidar, mask=io.imread(img_path), io.imread(lidar_path), io.imread(mask_path)

            data = [io.imread(dt) for dt in data]
            for j in range(num_hor):
                for k in range(num_ver):
                    start_x, end_x = j*pw, (j+1)*pw
                    start_y, end_y = k*ph, (k+1)*ph
                    
                    # only use rgb channel of img
                    img = data[0]
                    sub_img = img[start_x:end_x, start_y:end_y,0:3]
                    
                    # lidar img
                    lidar = data[1]
                    sub_lidar = lidar[start_x:end_x, start_y:end_y]

                    # only use gray scale mask
                    mask = data[2]
                    sub_mask = mask[start_x:end_x, start_y:end_y,1]
                    
                    file_name = "{}_{}_{}.png".format(i, j, k)

                    # save img
                    Image.fromarray(sub_img).save(
                        os.path.join(root_dir, config['proc_imgs'],file_name)
                        )

                    # save lidar
                    Image.fromarray(sub_lidar).save(
                        os.path.join(root_dir, config['proc_lidars'],file_name)
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
    data_type = kwargs['data_type']


    if data_type == 'rgb':
        # directory name to the processed data
        data_dir = config['proc_imgs']

        # filename of the mean and std json file
        mean_std_file = config['mean_std_rgb']
        num_channels = 3
    elif data_type == 'lidar':
        data_dir = config['proc_lidars']
        mean_std_file = config['mean_std_lidar']
        num_channels = 1
    else:
        raise("data_type can only be rgb or lidar")


    files = os.listdir(os.path.join(proj_root, data_dir))
    file_paths = [
        os.path.join(proj_root, data_dir, sgfile) 
        for sgfile in files]




    mean = np.zeros(num_channels)
    std = np.zeros(num_channels)
    for sgfile in file_paths:
        item = io.imread(sgfile).astype(np.float32)/255
        mean = mean + np.mean(item, axis=(0,1))

        std = std + np.std(item, axis=(0,1))
        print("std is:", std)

    mean = mean / len(files)
    std = std / len(files)

    # save mean and std to json
    mean_std = {
        "mean": np.ndarray.tolist(mean),
        "std": np.ndarray.tolist(std)
        }
    

    with open(os.path.join(proj_root, mean_std_file), 'w') as f:
        json.dump(mean_std, f)




    return
        
            
