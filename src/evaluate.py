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
        avg_acc = 0
        for step, (img, mask) in enumerate(loader):
            step = step + 1
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)

            acc = utils.pixelwise_accuracy(output, 
                    mask, threshold)
            avg_acc = avg_acc + acc
            
            msg = "Step : {}, Acc : {:0.3f}".format(step,acc)

            print(msg)
        avg_acc = avg_acc / step 
        print("Average acc : {:0.3f}".format(avg_acc))
