'''
Analyze training process and model performance
'''
import torch
from torch.utils.data import DataLoader
from model import Model, ModelV1
from dataset import TreeDataset, TreeDatasetV1


def _print(**kwargs):
    msg = ''
    for k, v in kwargs.items():
        msg = msg + '{} : {} '.format(k, v)
    print(msg)
    return

ckp = './elv_ckps/model_10.pth'

device = torch.device("cuda:0" if torch.cuda.is_available()
        else "cpu")

model = ModelV1()

model = model.load_state_dict(torch.load(ckp))
model = model.to(device)

print(model)

data = '/mnt/efs/Trees_processed'
ds = TreeDatasetV1(data, purpose='test')

dataloader = DataLoader(ds, batch_size=16, shuffle=False)

with torch.no_grad():
    for img, elv, mask in dataloader:
        img = img.to(device)
        mask = mask.to(device)
        
        output = model(img)
        tp, tn, fp, fn = _confusion_matrix(output, mask)
        _print(tp=tp, tn=tn, fp=fp, fn=fn)




