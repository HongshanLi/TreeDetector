import torch
import torch.nn as nn
import torchvision.models as models

class 4to3(nn.Module):
    '''Map 4 channels (RGBE) to 3 channels'''
    def __init__(self):
        super(RGBPlusElv, self).__init__()
        self.layers = nn.Sequential()
        conv1 = nn.Conv2d(4, 3, kernel_size=1,
                stride=1,padding=0)
        bn1 = nn.BatchNorm2d(3)
        act1 = nn.Sigmoid()

        self.layers.add_module('conv1', conv1)
        self.layers.add_module('bn1', bn1)
        self.layers.add_module('act1', act1)

    def forward(self, x):
        return self.layers(x)
        

class ResNetFE(nn.Module):
    '''Resnet backbone for feature extraction'''
    def __init__(self, pretrained=True):
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


class Model(nn.Module):
    '''final model for creating tree mask for 2d RGB images'''
    def __init__(self):
        super(Model, self).__init__()
        
        # feature extractor
        self.fe = ResNetFE()
        self.mask = Mask()
    
    def forward(self, x):
        x = self.fe(x)
        x = self.mask(x)
        return x

class ModelV1(nn.Module):
    '''use elvation image as an additional channel'''
    def __init__(self):
        super(ModelV1, self).__init__():
        self.4to3 = 4to3()
        self.3_channel_model = Model().load_state_dict(
            'ckps/model_20.pth')
    
    def forward(self, x):
        x = self.4to3(x)
        x = 3_channel_model(x)
        return x


if __name__=='__main__':
    m = Model()
    x = torch.Tensor(2, 3, 250, 250)
    y = m(x)
    print(y.shape)

