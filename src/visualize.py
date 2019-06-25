from torchviz import make_dot
from models import ResNetModel
import torch

def visualize_model():
    x = torch.zeros(1, 3, 250, 250)
    model = ResNetModel(pretrained=False)
    y = model(x)
    dg = make_dot(y)
    dg.render('test.gv', view=True)


if __name__=='__main__':
    visualize_model()
