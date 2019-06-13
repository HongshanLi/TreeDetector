import random
from utils import CocoYoLo
import torch
from torch.utils.data import DataLoader
import time
from darknet import Darknet
from trainer import Trainer

import streamlit as st


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
