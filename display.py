'''Display images and predicted masks using streamlit'''
import streamlit as st
import os
import argparse
import skimage.io as io
import src.utils as utils
import torch
import numpy as np

parser = argparse.ArgumentParser(
        description='Dispaly images and predicted masks')
parser.add_argument('--images', type=str, metavar='PATH',
        default='./static/imgs')
parser.add_argument('--lidars', type=str, metavar='PATH',
        default='./static/lidars')
parser.add_argument('--mask-dir-lidar', type=str, metavar='PATH',
        default='./static/resnet_mask_lidar')
parser.add_argument('--mask-dir', type=str,metavar='PATH',
        default='static/resnet_mask')
parser.add_argument('--target-dir', type=str, metavar='PATH',
        default='./static/targets', help='dir to hand labeled masks')

args = parser.parse_args()

def pixelwise_accuracy(mask, target):
    '''compute pixelwise accuracy
        Args:
            mask (np.float32): black and white mask
            black = object, white = background
            target (np.float32): ...
    '''
    correct = (mask == target).astype(np.float32)

    acc = np.sum(correct) / (mask.shape[0]*mask.shape[1])
    return acc.item()

def compute_iou(mask, target):
    '''compute intersection over union
        Args:
            mask (np.float32): black and white mask
            black = object, white = background
            target (np.float32): ...
    
    '''
    # make object have pixel value 1
    mask = (mask == 0).astype(np.float32)
    target = (target == 0).astype(np.float32)

    intersection = mask*target

    union = mask + target - intersection

    iou = np.sum(intersection) / np.sum(union)
    return iou.item()




def get_background(img):
    '''
    Args:
        img (np.uint8): input image
    Return (np.float32): mask of objects in the image
    white pixel for background, black pixel for non-background
    pixel value is normalized between [0, 1]
    '''
    img = img.astype(np.float32)
    img = np.mean(img, axis=2)
    img = img / 255
    img = (img == 1).astype(np.float32)
    return img

# names of imgs and masks
file_names = os.listdir(args.images)
images = [os.path.join(args.images, n) for n in file_names]
images = [io.imread(img) for img in images]

lidars = [os.path.join(args.lidars, n) for n in file_names]
lidars = [io.imread(lidar) for lidar in lidars]

masks = [os.path.join(args.mask_dir, n) for n in file_names]
masks = [io.imread(mask) for mask in masks]

masks_lidar = [os.path.join(args.mask_dir_lidar, n) for n in file_names]
masks_lidar = [io.imread(mask) for mask in masks_lidar]

targets = [os.path.join(args.target_dir, n) for n in file_names]
targets = [io.imread(mask) for mask in targets]


for img_name, img, lidar, mask, target in zip(file_names, images, lidars, masks, targets):
    st.write("Image name: ", img_name)

    st.image(image=[img, lidar, mask, target], 
            caption=["Input Image", "Lidar", "Predicted Mask", "True Mask"],
            width=150)


    mask = get_background(mask)
    target = get_background(target)

    pix_acc = pixelwise_accuracy(mask, target)
    iou = compute_iou(mask, target)

    st.write("Pixelwise accuracy: {:0.2f}".format(pix_acc) \
            + " Intersection over Union: {:0.2f}".format(iou))
    st.write('\n')
    


