import random
import torch
import time
from torch.utils.data import DataLoader

from dataset import TreeDataset
from dataset import TreeDatasetV1

from criterion import Criterion

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

loader = DataLoader(ds, batch_size=32, shuffle=False)
x, y, z = next(iter(loader))
c = Criterion()


print(z.shape)

num_tree, num_no_tree = c.find_pixels(z)
print(num_tree, num_no_tree)
