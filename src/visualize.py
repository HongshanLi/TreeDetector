import os
from torchviz import make_dot
from models import ResNetModel, UNet
import torch
import pickle
import matplotlib.pyplot as plt


def train_val_acc(logfile):
    with open(logfile, 'rb') as f:
        log=pickle.load(f)
    
    train_acc = []
    val_acc = []
    for i in range(len(log)):
        train_acc.append(log[i+1]['train_acc'])
        val_acc.append(log[i+1]['val_acc'])

    num_epochs = len(log)
    
    train_acc, = plt.plot(range(num_epochs), train_acc, "b")
    val_acc, = plt.plot(range(num_epochs), val_acc, "r")


    plt.legend([train_acc, val_acc], ["Train Accuracy", "Val Accuracy"])

    plt.savefig("./train_val_acc.png")



def visualize_model(model):
    x = torch.zeros(1, 3, 250, 250)
    y = model(x)
    dg = make_dot(y)
    file_name = model.model_name + "_model.gv"
    dg.render(os.path.join('../plots', file_name), view=False)
    return     

if __name__=='__main__':
    resnet=ResNetModel(pretrained=False)
    unet=UNet()
    visualize_model(resnet)
    visualize_model(unet)
