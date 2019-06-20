'''Baseline models'''
import argparse

import torch
from torch.utils.data import DataLoader
from dataset import TreeDataset
import utils
from model import Model


device = torch.device('cuda:0' if torch.cuda.is_available()
        else 'cpu')


parser = argparse.ArgumentParser(
        description='Baseline models to GreenNet')

parser.add_argument('--data', metavar='DIR',
    default='/mnt/efs/Trees_processed', 
    help='dir to processed data')
parser.add_argument('--model-ckp', metavar="PATH",
    default='./ckps/model_8.pth',
    help='path to model checkpoint')


args = parser.parse_args()

class Constant(object):
    '''constant classifier 
    Identifiy everything as bg
    '''
    def __call__(self, x):
        y = torch.ones_like(x).to(device)
        return y


if __name__=='__main__':
    ds = TreeDataset(args.data, purpose='test')
    dataloader = DataLoader(ds, batch_size=16)
    print("Size of the test data set:{}".format(len(ds)))
    print("Device: ", device)   
    baseline = Constant()

    model = Model()
    model = model.to(device)
    model.load_state_dict(
            torch.load(args.model_ckp, map_location=device))


    baseline_acc = 0
    model_acc = 0
    for step, (img, _, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)
        output = baseline(img)

        acc = utils.pixelwise_accuracy(output, mask)
        baselline_acc = baseline_acc + acc
        
        with torch.no_grad():
            output = model(img)
        acc = utils.pixelwise_accuracy(output, mask)
        model_acc = model_acc + acc
        

    baseline_acc = baseline_acc / (step + 1)
    model_acc = model_acc / (step + 1)
    
    msg ="Baseline acc: {}".format(baseline_acc)
    msg = msg + ' Model acc:{:0.2f}'.format(model_acc)
    print(msg)




