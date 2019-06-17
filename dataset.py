import imageio
import scipy
import PIL.Image as Image
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as ds
import torchvision.transforms as transforms

#from torchvision.datasets import VisionDataset
import numpy as np
import PIL.Image as Image
import os
import skimage.io as io
import random

img_dir='/home/hongshan/data/train2017'
annFile='/home/hongshan/data/annotations/instances_train2017.json'


# for the img, pad to 640 then centercrop to 608
# create the mask for the untransformed img, then
# pad to 640 and centercrop to 608


class CocoMask():
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
            os.makedirs(os.path.join(out_dir, 'elevations/'))
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
        
        # elvation img
        img_elv = img_id + "_DSM.tif"
        
        # mask
        img_mask = img_id + "_TREE.png"
        
        img_path = os.path.join(img_dir, 
                img_sfd, img_id, img_rgb)

        elv_path = os.path.join(img_dir,
                img_sfd, img_id, img_elv)

        mask_path = os.path.join(img_dir,
                img_sfd, img_id, img_mask)

        img = io.imread(img_path)
        elv = io.imread(elv_path)
        mask = io.imread(mask_path)
        
        # divid into subimgs and save 
        for i in range(num):
            for j in range(num):
                start_x, end_x = i*250, (i+1)*250
                start_y, end_y = j*250, (j+1)*250
                
                sub_img = img[start_x:end_x, start_y:end_y,0:3]

                # elv img only has one channel
                sub_elv = elv[start_x:end_x, start_y:end_y]

                sub_mask = mask[start_x:end_x, start_y:end_y,0:3]
                
                idx = '{}_{}'.format(i,j)
                file_name = img_id + '_' + idx + ".png"

                # save img
                Image.fromarray(sub_img).save(
                        os.path.join(out_dir, 'imgs/', file_name)
                        )

                # save elv
                Image.fromarray(sub_elv).save(
                        os.path.join(out_dir, 'elevations/', file_name)
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

class TreeDataset(Dataset):
    #TODO divid each img into 5 x 5 subimages
    # each of which has dimension 250 x 250 

    def __init__(self, proc_dir, transform=None, 
        elv_transform=None, mask_transform=None, purpose='train'):
        
        # get a list of img files
        self.proc_dir = proc_dir
        self.file_names = self.get_file_names()

        
    
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4137, 0.4233, 0.3968),
                (0.2275, 0.2245, 0.2424))
            ])

        self.elv_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])


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
        elv = os.path.join(self.proc_dir, 'elevations/', file_name)
        mask = os.path.join(self.proc_dir, 'masks/', file_name)

        img = io.imread(img)
        elv = io.imread(elv)
        mask = io.imread(mask)

        if self.transform is not None:
            img  = self.transform(img)

        if self.elv_transform is not None:
            elv = self.elv_transform(elv)
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            mask = mask[1,:,:].view(1,250,250)

        return img, elv, mask
    
    def get_file_names(self):
        file_names = os.listdir(os.path.join(
            self.proc_dir, 'imgs'))
        return file_names


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


    data_dir = '/mnt/efs/Trees_processed/'
    ds = TreeDataset(data_dir,purpose='validate')
    print(ds.file_names)

