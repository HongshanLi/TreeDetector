import random
from utils import CocoYoLo
import torch
from torch.utils.data import DataLoader
import time
from darknet import Darknet
from trainer import Trainer

#import streamlit as st

torch.set_num_threads(6)
img_dir = '/scr1/li108/data/coco_train_2017'
processed_label = 'processed/processed_0.pickle'
dset = CocoYoLo(img_dir, processed_label)
torch.autograd.set_detect_anomaly(True)
dark = Darknet('./cfg/yolov3.cfg', 'yolov3.weights')

tr = Trainer(dark, dset, 2, 1)

def main():
    tr.train_one_epoch()
    return


x = torch.randn(2, 3, 640, 640)
y = dark(x)
for k in dark.outputs:
    print(k, dark.outputs[k].shape)

z = dark.produce_mask()
print(z.shape)

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
