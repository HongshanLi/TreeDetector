import torch 
import torch.nn.functional as F

class Criterion(object):
    '''Loss function for mask
    Inplement focal loss to balance class 
    imbalance between tree and non-tree pixel
    '''
    def __call__(self, output, target):
        num_tree, num_bg = self.find_pixels(target)

        # target =1 => background
        # loss for miss-classified background
        bg = (output - target)*target
        bg = bg**2
        bg = torch.mean(bg)

        # loss for miss-classified tree
        tree = (output - target)*(1 - target)
        tree = tree**2
        tree = torch.mean(tree)

        # balance tree and bg pixels
        #loss = num_tree*bg + num_bg*tree
        #loss = torch.mean(loss)
        return bg, tree

    def find_pixels(self, target):
        '''compute the number of tree and non-tree pixels'''
        tree = (target == 1).float()
        num_tree = torch.sum(tree, dim=[1,2,3],keepdim=False)

        num_bg = torch.sum(target, dim=[1,2,3], keepdim=False)

        return num_tree, num_bg



