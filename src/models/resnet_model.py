import torch
import torch.nn as nn
import torchvision.models as models
from .post_process import PostProcess

class ResNetFE(nn.Module):
    '''Resnet backbone for feature extraction'''
    def __init__(self, pretrained):
        super(ResNetFE, self).__init__()
        self.resnet = models.resnet152(pretrained=pretrained)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class Mask(nn.Module):
    '''create mask of trees for each input img'''
    def __init__(self):
        super(Mask, self).__init__()
        # make things reproducible
        torch.manual_seed(8128)

        self.module_list = nn.ModuleList()
        for i in range(0, 5):
            if i <= 3:
                padding = 0
            else: 
                padding = 3

            in_channels = 2048 // (2**i)
            out_channels = 2048 // (2**(i+1))

            module = nn.Sequential()
            deconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2,
                stride=2, padding=padding)
            module.add_module('deconv_{0}'.format(i+1), deconv)
            
            bn = nn.BatchNorm2d(out_channels)
            module.add_module('bn_{}'.format(i+1), bn)

            act = nn.ReLU()
            module.add_module('relu_{0}'.format(i+1), act)
            
            self.module_list.append(module)

        in_channels = 2048 // (2**5)
        out_channels = 1

        mask = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                kernel_size=1),
            nn.Sigmoid()
            )

        self.module_list.append(mask)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class ResNetModel(nn.Module):
    '''final model for creating tree mask for 2d RGB images'''
    def __init__(self, pretrained, use_lidar):
        super(ResNetModel, self).__init__()
        self.model_name='resnet'

        self.use_lidar = use_lidar

        # feature extractor
        self.fe = ResNetFE(pretrained)
        self.mask = Mask()

        if self.use_lidar:
            self.post_process = PostProcess()
        
    
    def forward(self, x, lidar=None):
        x = self.fe(x)
        x = self.mask(x)
        
        if self.use_lidar:
            x = torch.cat([x, lidar], dim=1)
            x = self.post_process(x)
        return x




