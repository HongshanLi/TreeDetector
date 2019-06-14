import random
import torch
import time

from dataset import TreeDataset

def main():
    tr.train_one_epoch()
    return




def test_CocoMask(img_dir, annFile):
    '''Test the integrity of CocoMask class'''
    from dataset import CocoMask
    ds = CocoMask(img_dir, annFile, 'person')
    
    size=len(ds)
    # randomly select 1000 imgs
    for i in range(1000):
        idx=random.randint(0, size)
        im, mask = ds[idx]


#print(y.shape)
#print(dark)

data='/mnt/efs/Trees_processed/'
ds = TreeDataset(data)
x, y, z = ds[0]
print(z.shape, z)
