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
        avg_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for step, (img, mask) in enumerate(loader):
            step = step + 1
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)

            acc = utils.pixelwise_accuracy(output, 
                    mask, threshold)
            cm = utils.confusion_matrix(output, mask, 
                    threshold)
            avg_acc = avg_acc + acc

            for key in avg_cm.keys():
                avg_cm[key] = avg_cm[key] + cm[key]
            
            msg = "Step : {}, Acc : {:0.3f}".format(step,acc)
            print(msg)
        avg_acc = avg_acc / step 
        
        for key in avg_cm.keys():
            avg_cm[key] = avg_cm[key] / step
        print("Average acc : {:0.3f}".format(avg_acc))
        print(avg_cm)
