import torch 
import torch.nn.functional as F

class Criterion(object):
    '''Loss function for mask
    Inplement focal loss to balance class 
    imbalance between tree and non-tree pixel
    '''
    def __call__(self, output, target):
        loss = F.mse_loss(output, target)
        return loss

    def find_pixels(self, target):
        '''compute the number of tree and non-tree pixels'''
        tree = (target == 1).float()
        num_tree = torch.sum(tree, dim=[1,2,3],keepdim=False)
        num_bg = torch.sum(target, dim=[1,2,3], keepdim=False)
        return num_tree, num_bg



