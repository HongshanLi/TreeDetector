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



