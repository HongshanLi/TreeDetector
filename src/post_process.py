'''
Post process masks generated by the model
'''
import numpy as np

class CleanUp(object):
    '''classical CV algorithms to clean up the masks'''
    
    def __call__(self, x):
        x = self.suppression(x)
        return x

    def suppression(self, x, threshold=0.3):
        '''suppress pixel to 0 if it is less than <threshold>'''
        x = (x > threshold).astype(np.float32)
        return x

