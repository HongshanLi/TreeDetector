'''Display images and predicted masks using streamlit'''
import streamlit as st
import os
import argparse
import skimage.io as io

parser = argparse.ArgumentParser(
        description='Disply images and predicted masks')
parser.add_argument('--images', type=str, metavar='PATH',
        default='/mnt/efs/tmp_images')
parser.add_argument('--mask-dir', type=str, metavar='PATH',
        default='/mnt/efs/tmp_masks')
parser.add_argument('--gth-mask-dir', type=str, metavar='PATH',
        default='/mnt/efs/tmp_masks', help='dir to ground truth masks')

args = parser.parse_args()

# names of imgs and masks
file_names = os.listdir(args.images)
images = [os.path.join(args.images, n) for n in file_names]
images = [io.imread(img) for img in images]

masks = [os.path.join(args.mask_dir, n) for n in file_names]
masks = [io.imread(mask) for mask in masks]

gth_masks = [os.path.join(args.gth_mask_dir, n) for n in file_names]
gth_masks = [io.imread(mask) for mask in gth_masks]


for img_name, gth, img, mask in zip(file_names, gth_masks, images, masks):
    st.image([gth, img, mask], width=200)


