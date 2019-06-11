import torch
import torch.nn as nn
import torchvision.models as models

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
            in_channels = 2048 // (2**i)
            out_channels = 2048 // (2**(i+1))
            module = nn.Sequential()
            deconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3,
                stride=2, padding=0)
            module.add_module('deconv_{0}'.format(i+1), deconv)
            self.module_list.append(module)


    def forward(self, x):
        for module in self.module_list:
            x = module(x)
            print(x.shape)



        



if __name__=='__main__':
    m = Mask()
    x = torch.Tensor(2, 2048, 8, 8)
    y = m(x)

