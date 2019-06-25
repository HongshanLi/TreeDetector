import os
from torchviz import make_dot
from models import ResNetModel, UNet
import torch

def visualize_model(model):
    x = torch.zeros(1, 3, 250, 250)
    y = model(x)
    dg = make_dot(y)
    file_name = model.model_name + "_model.gv"
    dg.render(
            os.path.join('../plots/', file_name)
            )
    return     

if __name__=='__main__':
    resnet=ResNetModel(pretrained=False)
    unet=UNet()
    visualize_model(resnet)
    visualize_model(unet)
