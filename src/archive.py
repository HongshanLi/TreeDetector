'''unused code'''

class TreeDatasetV1(Dataset):
    '''stack elv image as an additional channel
    over rgb images'''
    def __init__(self, proc_dir, transform=None, 
        elv_transform=None, mask_transform=None, purpose='train'):
        
        # get a list of img files
        self.proc_dir = proc_dir
        self.file_names = self.get_file_names()
    
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4137, 0.4233, 0.3968),
                (0.2275, 0.2245, 0.2424))
            ])

        self.elv_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])


        # choose the same 90% imgs for training
        np.random.seed(42)
        total_size = len(self.file_names)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = int(total_size * 0.1)

        dist = []
        for i in range(total_size):
            if i < train_size:
                dist.append(0)
            elif i>= 1 and i <train_size + val_size:
                dist.append(1)
            elif i>= train_size + val_size and i < total_size:
                dist.append(2)

        train_set = []
        val_set = []
        test_set = []
        for i, b in enumerate(dist):
            if b == 0:
                train_set.append(self.file_names[i])
            elif b==1:
                val_set.append(self.file_names[i])
            elif b==2:
                test_set.append(self.file_names[i])

        if purpose=='train':
            self.file_names = train_set
        elif purpose=='val':
            self.file_names = val_set
        elif purpose=='test':
            self.file_names = test_set
        elif purpose==None:
            pass
        else:
            print("purpose must be 'train', 'val' or 'test'")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # img subfolder
        file_name = self.file_names[idx]

        img = os.path.join(self.proc_dir, 'imgs/', file_name)
        elv = os.path.join(self.proc_dir, 'elevations/', file_name)
        mask = os.path.join(self.proc_dir, 'masks/', file_name)

        img = io.imread(img)
        elv = io.imread(elv)
        mask = io.imread(mask)

        if self.transform is not None:
            img  = self.transform(img)

        if self.elv_transform is not None:
            elv = self.elv_transform(elv)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            mask = mask[1,:,:].view(1,250,250)
        
        img = torch.cat([img, elv], dim=0)
        return img, elv, mask
    
    def get_file_names(self):
        file_names = os.listdir(os.path.join(
            self.proc_dir, 'imgs'))
        return file_names


class CocoMask():
    """Create mask for a chosen category.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, catNm, transforms=None):
        super(CocoMask, self).__init__(root, transforms)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.catId = self.coco.getCatIds(catNms=[catNm])[0]

        #self.ids = list(sorted(self.coco.imgs.keys()))
        
        # only get images with person
        self.ids = self.coco.getImgIds(catIds=[self.catId])
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # select annotation whose catId is self.catId
        target =[]
        for ann in anns:
            if ann['category_id']==self.catId:
                target.append(ann)

        # create mask based on selected annotations
        path = coco.loadImgs(img_id)[0]['file_name']
        img = io.imread(os.path.join(self.root, path))
        
        h,w,c = img.shape
        mask = np.zeros((h,w))
        for ann in target:
            mask = mask + coco.annToMask(ann)

        mask = mask * 255
        mask = mask.astype(np.uint8)

        st.write("mask for img", mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, mask


    def __len__(self):
        return len(self.ids)

    def test(self):
        return

"""Preprocess the coco data for YOLO algorithm

This file takes the image info of dataset and create
data that can be readily used for training YOLO algorithm

Raw images in COCO dataset have different different 
dimensions, and there are also images that has only one
channel

The width and height of each image are less than or equal
to 640. So we pad every image to 640x640

If an image has only one channel, then we convert it
to a 3-channel image by repeating the same value for
each channel.

The output of this file is a collection of binary files
    data_chunk_i.pickle

Each output file is list of dictionary with keys:
    image: 
    label_1: label at scale 1
    label_2:
        

    label_n: label at scale n


The program follows this pipeline
    raw_img -> padded_img -> create_label_for_grid_cell


"""
import PIL
import numpy as np
import json
import argparse
import time
from pycocotools.coco import COCO
import pickle
from utils import compute_iou

import matplotlib.pyplot as plt
import matplotlib.patches as patches
parser = argparse.ArgumentParser(
        description="Preprocess COCO for YOLO algorithm")
parser.add_argument("--anns",type=str, dest='anns',
        default='/scr1/li108/data/annotations/instances_train2017.json',
        help="JSON file annotating instances in COCO images")
parser = parser.parse_args()



def label_grid(img_dim, annotations, grid_scheme, anchor_boxes):
    '''
    img_dim: dimension of the image to be feed into neural network
    annotaions: coco annotations of the img

    grid_scheme: how many grid cells to divide

    anchor boxes: list of tuples, each tuple specifies
    the width and height of one anchor box

    L = __len__(anchor boxes)
    
    The label of each grid cell consists of bbox corresponding
    to each anchor box. If each grid has L anchor boxes,
    then the label for each grid cell is a vector of length
    \[
        2 + 5*L
    \]
    It is of the form
    \[
        [i, j, k_1, b_k_1, c_k_1, ...]
    \]
    where i, j is the index of each grid, k_1 is the index of
    the first anchor that correspond to some bbox, c_k_1 is
    the class label of the object in that bbox

    At the end, only labels for the grid cell that contains
    the center of some bbox will be saved.
    '''
    nh, nv = grid_scheme
    w, h = img_dim

    grid_w = w // nh
    grid_h = h // nv


    def contains_center(bbox, i, j):
        '''
        given bbox, test if i,j grid cell
        contains its center
        '''
        # coordinate of center of bbox
        #bbox = tuple(bbox)
        x,y,w,h = tuple(bbox)
        #x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        center_x = x + (w // 2)
        center_y = y + (h // 2)

        gx, gy = i*grid_w, j*grid_h

        contains_x = (gx <= center_x <= gx + grid_w)
        contains_y = (gy <= center_y <= gy + grid_h)

        return contains_x*contains_y

    def get_abox_idx(bbox, i, j):
        '''
        given a grouth truth bbox,
        get the index of the anchor box at i,j grid cell
        that has the highest IOU
        '''
        ious=[]
        for anchor in anchor_boxes:
            anchor_coord = get_anchor(anchor, i, j)
            bbox = np.array(bbox)
            anchor_coord = np.array(list(anchor_coord))

            iou = compute_iou(bbox, anchor_coord)
            ious.append(iou)
        m = max(ious)
        return ious.index(m)

    def get_anchor(anchor, i, j):
        '''
        anchor: width and height of an anchor
        i, j: multi-index of grid

        compute the coordinate (x, y, w, h)
        of the anchor box given the width and height of
        the anchor and muti

        '''
        aw, ah = anchor
        # center of i, j grid
        center_x = i*grid_w + (grid_w // 2)
        center_y = j*grid_h + (grid_h // 2)

        # cooridnate of anchor box
        ax, ay = center_x - (aw // 2), center_y - (ah // 2)

        return (ax, ay, aw, ah)

    # initialze the label tensor
    # Y = torch.zeros(nh, nv, len(anchor_boxes), 6)
    labels = []
    for i in range(nh):
        for j in range(nv):
            # label for each grid
            label = np.zeros([3, 6], dtype=np.float16)
            for ann in annotations:
                bbox = ann['bbox']

                # weather grid i, j contains 
                # the center of bbox
                if contains_center(bbox, i, j):
                    # FIXME one anchor box could have 
                    # highest IOU with multiple bbox
                    aidx = get_abox_idx(bbox, i, j)
                    info = [1] + bbox + [ann['category_id']] 
                    info = np.array(info)
                    label[aidx] = info
            # check if to save the label
            if label.sum() != 0:
                label = label.flatten('C')
                grid_idx = np.array([i, j],dtype=np.float16)
                label = np.concatenate([grid_idx, label], axis=0)
                labels.append(label)

    try:
        labels = np.stack(labels)
    except Exception as e:
        print(e)
        print(annotations, grid_scheme)


    return labels


def show_plot():
    '''plot Y
    Used for debugging
    test if this function does what it supposes to do
    plot img, plot bbx, plot anchor box with the
    highest iou with the bbx
    '''
    if isinstance(img, PIP.Image.Image):
        img_ = np.array(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        img_ = img

    # For each grid if it has object then plot the bbox
    # for that object and the anchor box responsible for
    # detecting that object

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            a0 = Y[i, j, 0]
            a1 = Y[i, j, 1]
            for k in range(Y.shape[2]):
                if Y[i, j, k, 0].item() == 1.0:
                    fig,ax=plt.subplots(1)
                    ax.imshow(img_)

                    # plot grid boundary
                    gx,gy,gw,gh = i*grid_w, j*grid_h, grid_w, grid_h

                    rect=patches.Rectangle((gx,gy),gw,gh,
                                          linewidth=1,
                                          edgecolor='black',
                                          facecolor='none')
                    ax.add_patch(rect)


                    # plot bbox
                    bbox = Y[i, j, k,1:5].numpy()
                    bbox = tuple(bbox)
                    bx,by,bw,bh = bbox
                    rect=patches.Rectangle((bx,by),bw,bh,linewidth=1,
                         edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                    # plot anchor
                    achx,achy,achw,achh=get_anchor(anchor_boxes[k],i,j)
                    rect=patches.Rectangle((achx,achy),achw,achh,
                                           linewidth=1,
                         edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)

                    plt.show()

        return


def divide_imgs(img_dir, sub_img_dim):
    '''divide HD images (1250x1250) into subimages'''



def main():
    coco = COCO(parser.anns)
    ids = list(sorted(coco.imgs.keys()))
    start = time.time()
    
    '''
    process one chunk at a time
    so that even if some code 
    does not work, I don't have to
    wait until it processed all data
    '''
    chunks = (len(ids) // 1000) + 1
    f = open("unannotated_imgs", "w")
    for chunk in range(chunks):
        processed = []
        for i in range(chunk*1000, (chunk+1)*1000):
            img_label = {}
            img_id = ids[i]

            img_label['image_id'] = img_id
            img_label['file_name']=coco.loadImgs(img_id)[0]['file_name']

            ann_ids = coco.getAnnIds(imgIds=img_id)
            y = coco.loadAnns(ann_ids)

            # some images are not annotated
            if len(y) == 0:
                msg="Image with id: {} does not have annotations\n".format(
                    str(img_id))
                continue

            scale_1 = label_grid(img_dim=(640,640), annotations=y, 
                grid_scheme=(20,20),
                anchor_boxes=[(116,90), (159,198),(373, 326)])    
            scale_2 = label_grid(img_dim=(640,640), annotations=y, 
                grid_scheme=(40,40),
                anchor_boxes=[(30,61), (62,45),(59,119)])
            scale_3 = label_grid(img_dim=(640,640), annotations=y, 
                grid_scheme=(80,80),
                anchor_boxes=[(10,13), (16,30),(33, 23)])

            img_label['scale_1'] = scale_1
            img_label['scale_2'] = scale_2
            img_label['scale_3'] = scale_3

            processed.append(img_label)
            
        process_file = 'processed_{}.pickle'.format(chunk)
        with open(process_file, 'wb') as f:
            pickle.dump(processed, f)
    f.close()
    end = time.time()
    print(end - start)
    return

if __name__=="__main__":
    main()
    st.image(mask[:,:,:4])
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from utils import *
import cv2

def get_test_input():
    img = cv2.imread("./images/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']  
    lines = [x.rstrip().lstrip() for x in lines]
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


# create a nn.Module for each block in blocks

def create_modules(blocks):
    net_info = blocks[0]     
    module_list = nn.ModuleList()
    index = 0    
    prev_filters = 3
    output_filters = []
    for x in blocks:
        module = nn.Sequential()
        
        if (x["type"] == "net"):
            continue
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
                
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
                
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
            
        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = Interpolate(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
        
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            
            #Start  of a route
            start = int(x["layers"][0])
            
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            
            #Positive anotation
            if start > 0: 
                start = start - index
            
            if end > 0:
                end = end - index

            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            
            
            
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            
            module.add_module("maxpool_{}".format(index), maxpool)
        
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        else:
            print("Something I dunno")
            assert False

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
    return (net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest', 
            size=None, align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                mode=self.mode, align_corners=None)
        return x
    
class Mask(nn.Module):
    def __init__(self, input_size):
        super(Mask, self).__init__()
        self.input_size=input_size
        # upsample by 2
        kernel_size = 3
        pad = (kernel_size-1)//2
        
        self.conv1 = nn.ConvTranspose2d(
                256, 128, kernel_size, 2, pad)
        self.conv2 = nn.ConvTranspose2d(
                128, 56, kernel_size, 2, pad)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        

class Darknet(nn.Module):
    def __init__(self, cfgfile, weightsfile):
        super(Darknet, self).__init__()
        torch.manual_seed(9999)
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.load_weights(weightsfile)
        #self.replace_conv_layer()
        self.outputs = {}
        self.create_mask_layer()
    def replace_conv_layer(self):
        '''replace the conv layer before yolo layer
        The conv layer before yolo layer output a tensor
        of shape B, gs, gs, 255
        255 = num_anchor * (5 + num_classes)
        in their implementation the used only 80 classes 
        from COCO. But I want to use 90 classes. 
        So I need to replace it by a conv layer that outputs
        a tensor of shape
        B, gs, gs, 285
        '''
        for i, module in enumerate(self.blocks[1:]):
            if module['type'] == 'yolo':
                print("Replace this conv layer:", self.module_list[i-1])
                
                old_module = self.module_list[i-1]
                name = 'conv_{0}'.format(i-1)

                old_conv = old_module._modules[name]
                
                new_module = nn.Sequential()
                new_conv = nn.Conv2d(old_conv.in_channels, 285, 
                        old_conv.kernel_size, old_conv.stride)
                
                new_module.add_module(name, new_conv)
                self.module_list[i-1] = new_module
                print("Replaced by:", self.module_list[i-1])

    def create_mask_layer(self):
        '''create a mask layer after feature extraction'''
        self.mask=nn.Sequential()
        mask = Mask(603)
        self.mask = mask
        return

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {} # cache the output from route layer
        write = 0
        
        detections = []
        
        for i, module in enumerate(modules):
            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                        
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    
                    
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
            
            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x
                
            
            
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
   
                
                #Get the number of classes
                num_classes = int (modules[i]["classes"])
                
                x = x.view(x.shape[0], -1, 3, 95)
                
                # squeeze the object conf and 
                # coordinate of bbox prediction to [0,1]
                # expontiate the predict bbox height and weight
                # and multiply it by anchor box dimension
                x[:, :, :, 0:5] = torch.sigmoid(x[:,:,:,0:5])

                anchors = torch.Tensor(anchors)
                x[:,:,:,3:5] = torch.exp(x[:,:,:,3:5])*anchors

                # apply softmax to the object class
                #x[:,:,:,5:]=torch.softmax(x[:,:,:,5:],dim=3)
                detections.append(x)
                outputs[i] = outputs[i-1]
        try:
            self.outputs = outputs
            detections = torch.cat(detections, dim=1)
            return detections
        except Exception as e:
            print(e)
            return 0
        
    def produce_mask(self):
        x = self.outputs[104]
        #@TODO delete print statement
        print('feature shape:', x.shape)
        return self.mask(x)

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
    
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias)
                    bn_weights = bn_weights.view_as(bn.weight)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



import streamlit as st
import matplotlib.pyplot as plt
import os
import numpy as np


d = '/home/hongshan/data/Trees/037185/037185-0'
rgba = "037185-0_RGB-Ir.tif"
dsm = '037185-0_DSM.tif'

epth = os.path.join(d, dsm)

rgbpth = os.path.join(d, rgba)

b = plt.imread(rgbpth)
c = b[:,:,0:2]
a = plt.imread(epth)
a_ = a.reshape(1250, 1250, 1)

c_ = np.concatenate([c, a_], axis=2)


st.image(c_)
st.image(a_)







