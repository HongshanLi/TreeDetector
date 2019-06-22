import torch 

device = torch.device("cuda:0" if torch.cuda.is_available()
        else "cpu")
def confusion_matrix(output, target, threshold=0.5):
    '''confusion matrix for binary classfier'''

    # In target, 0 means tree 1 means bg
    predicted_mask = (output > threshold).float()

    # Positive = Tree pixel
    tp = (predicted_mask == 0).float()*(1 - target)
    tp = torch.sum(tp)
    p = (target == 0).float()
    p = torch.sum(p)
    tp = tp / p

    tn = (predicted_mask == 1).float()*target
    tn = torch.sum(tn)
    n = (target == 1).float()
    n = torch.sum(n)
    tn = tn / n

    fp = (predicted_mask == 0).float()*target
    fp = torch.sum(fp)
    fp = fp / n

    fn = (predicted_mask == 1).float()*(1 - target)
    fn = torch.sum(fn)
    fn = fn / p
    
    cm = {"TP": tp, "TN": tn, "FP": fp, "FN":fp}
    for key in cm.keys():
        cm[key] = cm[key].cpu().item()

    return cm

    

    
def pixelwise_accuracy(output, target, threshold=0.5):
    '''
    Args: 
        output: predicted mask
        target: true mask
        threshold: threshold to turn softmax into one-hot encode
    '''
    predicted_mask = (output > threshold).float()
    correct = (predicted_mask == target).float()
    acc = torch.sum(correct) / torch.sum(torch.ones_like(target))
    acc = acc.cpu().item()

    return acc
