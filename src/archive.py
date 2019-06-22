

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
