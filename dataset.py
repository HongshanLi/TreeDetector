from torch.utils.data import Dataset
import torchvision.datasets as ds
import torchvision.transforms as transforms

from torchvision.datasets import VisionDataset
import numpy as np
import PIL.Image as Image
import os
import skimage.io as io
import streamlit as st
import random

img_dir='/home/hongshan/data/train2017'
annFile='/home/hongshan/data/annotations/instances_train2017.json'


# for the img, pad to 640 then centercrop to 608
# create the mask for the untransformed img, then
# pad to 640 and centercrop to 608


class CocoMask(VisionDataset):
    """Create mask for a chosen category.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, catNm, transforms=None):
        super(CocoMask, self).__init__(root, transforms)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.catId = self.coco.getCatIds(catNms=[catNm])[0]

        #self.ids = list(sorted(self.coco.imgs.keys()))
        
        # only get images with person
        self.ids = self.coco.getImgIds(catIds=[self.catId])
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # select annotation whose catId is self.catId
        target =[]
        for ann in anns:
            if ann['category_id']==self.catId:
                target.append(ann)

        # create mask based on selected annotations
        path = coco.loadImgs(img_id)[0]['file_name']
        img = io.imread(os.path.join(self.root, path))
        
        h,w,c = img.shape
        mask = np.zeros((h,w))
        for ann in target:
            mask = mask + coco.annToMask(ann)

        mask = mask * 255
        mask = mask.astype(np.uint8)

        st.write("mask for img", mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, mask


    def __len__(self):
        return len(self.ids)

    def test(self):
        return

class TreeDataset(Dataset):
    #TODO divid each img into 5 x 5 subimages
    # each of which has dimension 250 x 250 

    def __init__(self, img_dir, train=True):
        # get a list of img ids
        self.img_dir = img_dir
        self.img_ids = self.create_img_ids()

        # choose the same 90% imgs for training
        np.random.seed(42)
        total_size = len(self.img_ids)
        is_train = np.random.binomial(size=total_size, n=1, p=0.9)
        
        train_ids = []
        val_ids = []
        for i, b in enumerate(is_train):
            if b == 1:
                train_ids.append(self.img_ids[i])
            else:
                val_ids.append(self.img_ids[i])

        if train:
            self.img_ids = train_ids
        else:
            self.img_ids = test_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # img subfolder
        img_id = self.img_ids[idx]

        img_sfd = img_id.split('-')[0]
        img_rgb = img_id + "_RGB-Ir.tif"
        img_mask = img_id + "_TREE.png"
        
        img_path = os.path.join(self.img_dir, img_sfd, img_id, img_rgb)
        mask_path = os.path.join(self.img_dir, img_sfd, img_id, img_mask)

        img = io.imread(img_path)
        mask = io.imread(mask_path)
        return img, mask


    def create_img_ids(self):
        '''create imgids based on their directory names'''
        img_ids = []
        for x in os.listdir(self.img_dir):
            img_ids = img_ids + os.listdir(
                    os.path.join(self.img_dir, x))
        return img_ids

        

if __name__=='__main__':
    img_dir = '/home/hongshan/data/Trees'
    ds = TreeDataset(img_dir)
    img, mask = ds[0]

    st.image(img[:,:,:3])

    st.image(mask[:,:,:4])


