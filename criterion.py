import torch 
import torch.nn.functional as F

class Criterion(object):

    def __call__(self, output, target):
        return F.mse_loss(output, target)


