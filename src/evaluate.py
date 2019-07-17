'''Evaluate model performance on test set'''
import torch
from torch.utils.data import DataLoader
import utils
import time

def evaluate_model(test_dataset, model, **kwargs):
    device = kwargs['device']
    batch_size = kwargs['batch_size']
    threshold=kwargs['threshold']

    loader = DataLoader(test_dataset, 
            batch_size=batch_size,
            shuffle=False)
    
    with torch.no_grad():
        avg_acc = 0
        avg_iou = 0
        avg_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for step, (img, lidar, mask) in enumerate(loader):
            step = step + 1    

            start = time.time()
            img = img.to(device)
            mask = mask.to(device)
            lidar=lidar.to(device)

            output = model(img, lidar)
            end = time.time()

            
            # Pixel Acc
            acc = utils.pixelwise_accuracy(output, 
                    mask, threshold)
            avg_acc = avg_acc + acc

            # IOU
            iou = utils.iou(output, mask, threshold)
            avg_iou = avg_iou + iou

            # Confusion matrix
            cm = utils.confusion_matrix(output, mask, 
                    threshold)
            for key in avg_cm.keys():
                avg_cm[key] = avg_cm[key] + cm[key]
            
            msg = "Step : {}, Acc : {:0.3f}".format(step, acc)
            msg = msg + " IOU : {:0.3}".format(iou)
            msg = msg + " Speed: {:0.2f} imgs/ sec".format(
                    img.shape[0] / (end - start) )
            print(msg)
        
        avg_acc = avg_acc / step 
        avg_iou = avg_iou / step 

        for key in avg_cm.keys():
            avg_cm[key] = avg_cm[key] / step
        print("Average acc : {:0.3f}".format(avg_acc))
        print("Average iou : {:0.3f}".format(avg_iou))
        print("confusion matrix", avg_cm)



