'''
Post process with lidar image
'''
import torch.nn as nn

class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

        
        block = nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU())

        layers = []
        for i in range(1):
            layers.append(block)
        
        self.layers = nn.Sequential(*layers)

        self.mask = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid())

    def forward(self, x):
        x = self.layers(x)
        x = self.mask(x)
        return x
