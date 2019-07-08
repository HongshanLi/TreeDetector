'''
Post process masks generated by the model
'''
import numpy as np

class CleanUp(object):
    '''classical CV algorithms to clean up the masks'''
    def __init__(self, threshold):
        '''
        Args: 
            threshold: threshold value to decide if a pixel
            should be white or black
        '''
        self.threshold = threshold

    
    def __call__(self, x):

        x = self.suppression(x)
        pretty_mask = self.beautify(x)
        return pretty_mask

    def suppression(self, x):
        '''
        a pixel is white (background) if and only if its value is
        bigger than threshold
        '''
        x = (x > self.threshold).astype(np.float32)
        return x


    def beautify(self, x):
        '''make black pixel green'''
        # create R channel
        r = (x == 0).astype(np.float32)
        r = r * 0.33
        r = x + r
        r = np.expand_dims(r, axis=0)

        # create G channel
        g = (x == 0).astype(np.float32)
        g = g * 1.0
        g = x + g
        g = np.expand_dims(g, axis=0)
        
        # create B channel
        b = x
        b = np.expand_dims(b, axis=0)

        pretty_mask = np.concatenate([r,g,b], axis=0)
        pretty_mask = np.transpose(pretty_mask, axes=[1,2,0])
        return pretty_mask
