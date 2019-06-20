import torch 

device = torch.device("cuda:0" if torch.cuda.is_available()
        else "cpu")
def confusion_matrix(output, target, threshold=0.5):
    '''confusion matrix for binary classfier'''

    # In target, 0 means tree 1 means bg
    predicted_mask = (output > threshold).float()

    # Positive = Tree pixel
    tp = (predicted_mask == 0).float()*(1 - target)
    tp = torch.sum(tp, dim=[1,2,3], keepdim=False)

    tn = (predicted_mask == 1).float()*target
    tn = torch.sum(tn, dim=[1,2,3], keepdim=False)

    fp = (predicted_mask == 0).float()*target
    fp = torch.sum(fp, dim=[1,2,3], keepdim=False)

    fn = (predicted_mask == 1).float()*(1 - target)
    fn = torch.sum(fn, dim=[1,2,3], keepdim=False)

    total = torch.ones(output.shape[0]).fill_(250*250).to(device)

    tp = torch.mean(tp / total)
    tn = torch.mean(tn / total)
    fp = torch.mean(fp / total)
    fn = torch.mean(fn / total)
    
    cm = [tp, tn, fp, fn]
    for i in range(4):
        cm[i] = cm[i].cpu().item()

    return tuple(cm)

    
def pixelwise_accuracy(output, target, threshold=0.5):
    '''
    Args: 
        output: predicted mask
        target: true mask
        threshold: threshold to turn softmax into one-hot encode
    '''
    predicted_mask = (output > threshold).float()
    correct = (predicted_mask == target).float()

    acc = torch.sum(correct) / torch.sum(torch.ones(target.shape))
    acc = acc.cpu().item()

    return acc
