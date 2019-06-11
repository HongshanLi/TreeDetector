from darknet import Darknet
from utils import CocoYoLo
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


class Trainer():
    def __init__(self, model, dataset, batch_size, epochs):
        self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epochs = epochs
        self.dataloader = DataLoader(self.dataset, batch_size, 
                shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters())

    def train_one_epoch(self):
        for step, (img, target) in enumerate(self.dataloader):
            start = time.time()
            img = img.to(self.device)
            target = target.to(self.device).float()
            y = self.model(img)
            loss = self.compute_loss(y, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            end = time.time()
            run = end - start
            print("Step: {}, Loss: {}, Run time: {}".format(
                step, loss.detach().cpu().item(), run))
    
    def compute_loss(self, output, target):
        '''
        most anchors will not be responsible
        for detection
        adjust weight for object confidence 
        loss
        '''
        noobj = 0.1

        # idx of anchors that responsible for some bbox
        # active anchor idx
        aai = (target[:,:,:,0] == 1.0)
        
        # idx of anchors not used for detection
        # inactive anchor idx
        iai = (target[:,:,:,0] == 0.0)
        
        # has object
        output_o = output[aai][:,0]
        target_o = target[aai][:,0]
        loss_has_object = F.mse_loss(output_o, target_o)

        # has no object
        output_no = output[iai][:,0]
        target_no = target[iai][:,0]
        loss_no_object = noobj*F.mse_loss(output_no, target_no)

        # x coordinate of predicted bbox center 
        # offset by the current grid location
        output_x = output[aai][:,1]
        target_x = target[aai][:,1]
        loss_x = F.mse_loss(output_x, target_x)

        # y coordiate of predicted bbox center
        # offset by the current grid location
        output_y = output[aai][:,2]
        target_y = target[aai][:,2]
        loss_y = F.mse_loss(output_y, target_y)

        # predicted / target bbox width
        output_w = output[aai][:,3]
        target_w = target[aai][:,3]
        loss_w = F.mse_loss(output_w, target_w)

        # predicted / target bbox height
        output_h = output[aai][:,4]
        target_h = output[aai][:,4]
        loss_h = F.mse_loss(output_h, target_h)

        # object class
        output_c = output[aai][:,5:]
        # coco annotation catogory id is in [1, 80]
        target_c = target[aai][:,5].long() - 1
        
        try:
            loss_class = F.cross_entropy(output_c, target_c)
        except Exception as e:
            print(e)
            print(output_c, target_c)

        loss = [loss_has_object, loss_no_object, 
                loss_x, loss_y, loss_w, loss_h,
                loss_class]
        loss = sum(loss)

        return loss

    def debug(self):
        img, target = next(iter(self.dataloader))
        img = img.to(self.device)
        target = target.to(self.device).float()
        y = self.model(img)

        loss = self.compute_loss(output=y, target=target)
        print(loss)


