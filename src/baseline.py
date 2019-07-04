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


class PixelThreshold(object):
    '''Threshold on Green pixels
    Identify green pixels as trees
    '''
    def __init__(self, threshold):
        '''
        Args:
            threshold: threshold for green pixel.
            If green channel of a pixel in image tensor 
            is bigger than [threshold], identify it as
            tree
        '''
        self.threshold = threshold
    def __call__(self, imgs):
        green_channel = imgs[:,1,:,:]
        # white pixels are background
        mask = green_channel < self.threshold
        return mask



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

    baseline_cm = {'TP':0, 'TN':0, 'FP': 0, 'FN':0}

    model_cm = {'TP':0, 'TN':0, 'FP': 0, 'FN':0}
    for step, (img, _, mask) in enumerate(dataloader):
        output = baseline(mask)

        acc = utils.pixelwise_accuracy(output, mask)
        baseline_acc = baseline_acc + acc
        
        cm = utils.confusion_matrix(output, mask)

        for key in cm.keys():
            baseline_cm[key] = baseline_cm[key] + cm[key]
        
        with torch.no_grad():
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)

        acc = utils.pixelwise_accuracy(output, mask)
        model_acc = model_acc + acc
        cm = utils.confusion_matrix(output, mask)

        for key in cm.keys():
            model_cm[key] = model_cm[key] + cm[key]

    baseline_acc = baseline_acc / (step + 1)
    model_acc = model_acc / (step + 1)

    for key in cm.keys():
        baseline_cm[key] = baseline_cm[key] / (step + 1)
        model_cm[key] = model_cm[key] / (step + 1)

    print("Baseline performance:")
    print("Acc: {:0.2f}".format(baseline_acc))
    for key in baseline_cm.keys():
        print('{} : {:0.2f}'.format(key, baseline_cm[key]))

    print("Model performance:")
    print("Acc: {:0.2f}".format(model_acc))
    for key in model_cm.keys():
        print('{} : {:0.2f}'.format(key, model_cm[key]))
    

    




