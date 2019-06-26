import torch 

device = torch.device("cuda:0" if torch.cuda.is_available()
        else "cpu")

def confusion_matrix(output, target, threshold=0.5):
    '''confusion matrix for binary classfier'''
    # In target, 0 means tree 1 means bg
    predicted_mask = (output > threshold).float()

    # Positive = Tree pixel

    # true positive 
    p = (target == 0).float()
    tp = (predicted_mask == 0).float()*p
    tp = torch.sum(tp)
    tp = tp / torch.sum(p)
    
    # true negative
    n = (target == 1).float()
    tn = (predicted_mask == 1).float()*n
    tn = torch.sum(tn)
    tn = tn / torch.sum(n)

    # false positive
    fp = (predicted_mask == 0).float()*n
    fp = torch.sum(fp)
    fp = fp / torch.sum(n)

    # false negative
    fn = (predicted_mask == 1).float()*p
    fn = torch.sum(fn)
    fn = fn / torch.sum(p)
    
    cm = {"TP": tp, "TN": tn, "FP": fp, "FN":fn}
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
