'''Evaluate model performance on test set'''
import torch
from torch.utils.data import DataLoader
import utils

def evaluate_model(test_dataset, model, **kwargs):
    device = kwargs['device']
    batch_size = kwargs['batch_size']
    threshold=kwargs['threshold']

    loader = DataLoader(test_dataset, 
            batch_size=batch_size,
            shuffle=False)
    
    with torch.no_grad():
        for step, (img, mask) in enumerate(loader):
            step = step + 1
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)

            acc = utils.pixelwise_accuracy(output, 
                    target, threshold)
            
            msg = "Step : {}, Acc : {}".format(step,acc)

            print(msg)
