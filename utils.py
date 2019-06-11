from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from torch.autograd import Variable
import numpy as np
import cv2 
import pickle
import PIL
import PIL.Image as Image
import os

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    
    grid_size = prediction.size(2)
    stride =  inp_dim // prediction.size(2)    
      
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
  
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, 
                                 grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, 
                                 grid_size*grid_size*num_anchors, 
                                 bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    

    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    #x_y_offset means the left-right cooridate
    # of each grid
    x_y_offset = torch.cat(
        (x_offset, y_offset), 1).repeat(
        1,num_anchors).view(-1,2).unsqueeze(0)
    
    
    prediction[:,:,:2] += x_y_offset 
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
   
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #push the class score to 0, 1 
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((
        prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    return prediction



def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False


    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

		
        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def letter_image(img, inp_dim):
    '''resize image with unchanged aspect
    ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))

    resized_image = cv2.resize(img, (new_w, new_h),
            interpolate=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1]. inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    '''
    Prepare image for inputting to the neural net
    '''
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(225.0).unsqueeze(0)
    return img

def compute_iou(rec1, rec2):
    '''
    Given two rectangles rec1 and rec2 
    computes its iou
    
    Args:
        rec1: numpy array of length 4
        containing:bx, by, bw, bh:
        the top left coordiate, width and height of 
        the rectangle

    '''
    for rec in [rec1, rec2]:
        if not isinstance(rec, np.ndarray):
            raise("arguments must be 1d numpy array of length 4" + \
                    "got {} and {}".format(rec1, rec2))
    
    
    # get coordinate of vertices of rec1 and rec2
    # in the order top-left, top-right, bottom-left, bottom-right
    # write it as a tensor of shape B x 4 x 2
    def get_coordinate(rec):
        '''get the coordinate of 4 vetices of of the rec
        The order of the vertices is
        top-left, top-right, bottom-left, bottom-right
        '''

        w = rec[2]
        # w is only added to the x cooridate
        w = w.repeat(repeats=2, axis=0)
        w[1] = 0    
        
        h = rec[3]
        # h is only added to th y cooridate
        h = h.repeat(repeats=2, axis=0)
        h[0] = 0
        
        top_left = rec[0:2]
        top_right = top_left + w
        bottom_left = top_left + h
        bottom_right = bottom_left + w
        
        coords = [top_left, top_right, 
                 bottom_left, bottom_right]
        return np.stack(coords)
    
    coord_1 = get_coordinate(rec1)
    #print("coordinate 1", coord_1)
    coord_2 = get_coordinate(rec2)
    #print("coordindate 2", coord_2)
    
    
    # compute coordinate for the intersection
    left_edge_x = np.maximum(coord_1[0, 0], coord_2[0,0])
    #print("left_edge_x", left_edge_x)
    right_edge_x = np.minimum(coord_1[1,0], coord_2[1,0])
    #print("right_edge_x", right_edge_x)
    top_edge_y = np.maximum(coord_1[0,1], coord_2[0,1])
    #print('top_edge_y', top_edge_y)
    bottom_edge_y = np.minimum(coord_1[2,1], coord_2[2,1])
    #print('bottom_edge_y', bottom_edge_y)
    
    intersection = (right_edge_x - left_edge_x)*(bottom_edge_y - top_edge_y)
    
    #print("intersection is", intersection)
    union = rec1[2]*rec1[3] + rec2[2]*rec2[3] - intersection
    
    return max(0, float(intersection) / float(union)) 


class PadImage(object):
    def __call__(self, img):
        if not isinstance(img, PIL.Image.Image):
            raise TypeError("input needs to be a PIL image")

        w, h = img.size

        if w > 640 or h > 640:
            print("oversized image")

        try:
            img_ = np.array(img, dtype=np.uint8)
            img_ = img_.transpose(1, 0, 2)
        except ValueError:
            # might be a gray scale image
            try:
                img_ = np.expand_dims(img_, axis=2)
                img_ = img_.repeat(3, axis=2)
            except ValueError:
                print("cannot transpose a tensor of shape {}".format(
                    img_.shape))

        #pad width
        try:
            pw = 640 - img_.shape[0]
            pw = np.zeros([pw, h, 3], dtype=np.uint8)
            img_ = np.concatenate([pw, img_], axis=0)
        except ValueError:
            print("image size:", img_.shape)
            print('concate to', pw.shape)


        #pad height
        try:
            ph = 640 - img_.shape[1]
            ph = np.zeros([img_.shape[0], ph, 3], dtype=np.uint8)
            img_ = np.concatenate([img_, ph], axis=1)
        except ValueError:

            print("image size: ", img_.shape)
            print("concate to", ph.shape)

        return img_.transpose(1,0,2)

    def __repre__(self):
        return self.__class__.__name__+'()'


class CocoYoLo(Dataset):
    '''Dataset for training YOLO algorithm on COCO images'''
    def __init__(self, image_dir, processed_label, transform=None):
        # use coco api to load image
        # open the processed label files 
        # only open one processed label at this point
        self.img_dir=image_dir

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                PadImage(),
                transforms.ToTensor()
                ])

        with open(processed_label, 'rb') as f:
            self.target = pickle.load(f)
        
    def __len__(self):
        return len(self.target)
    
    def transform_target(self,grid_label, grid_scheme, anchors):
        '''Transform the lable to be used for network
        grid_scheme: scale of yolo detection layer
        grid_label: processed label for <grid_scheme>
        anchors: anchors of the <grid_scheme>
        '''
        nh, nv = grid_scheme
        assert(nh == nv)

        img_dim = 640

        # grid size
        gs = img_dim // nh

        # initialize the desired label
        L = np.zeros([nh*nv, 3, 6])
        for each in grid_label:
            i = int(each[0].item())
            j = int(each[1].item())
            gl = each[2:].reshape(3, 6)
            # replace the top left coordinate of bbox
            # by its center relative to the grid cell
            # then normalize it between [0, 1]
            # center of each bbox
            cxy = gl[:,1:3] + (gl[:,3:5] / 2)

            # relative to the grid 
            grid_coord = np.array([i*gs, j*gs])
            rcxy = cxy - grid_coord

            # normalize 
            rcxy = rcxy / gs

            gl[:,1:3] = rcxy

            # Let tw be the output of the network
            # representing the width of the bbox bw
            # then one should have
            # bw = aw * e^{tx}
            # where aw means the width of the anchor 
            # responsible for that detection
            
            
            L[i*j] = gl
            

        return L.reshape(-1, 3, 6)
            
    
    def __getitem__(self, idx):
        file_name = self.target[idx]['file_name']
        img = Image.open(
                os.path.join(self.img_dir, file_name)).convert("RGB")
        
        t1 = self.target[idx]['scale_1']
        t2 = self.target[idx]['scale_2']
        t3 = self.target[idx]['scale_3']
        
        img = self.transform(img)
        
        # anchors to the first scale
        a1 = np.array([[116,90],[159,198],[373,326]], dtype=np.float16)
        t1 = self.transform_target(t1, (20, 20), a1)

        a2 = np.array([[30,61],[62,45],[59,119]],dtype=np.float16)
        t2 = self.transform_target(t2, (40, 40), a2)

        a3 = np.array([[10, 13],[16,30],[33,23]],dtype=np.float16)
        t3 = self.transform_target(t3, (80, 80), a3)
        
        target = torch.from_numpy(np.concatenate([t1,t2,t3],axis=0))
        return img, target

    
