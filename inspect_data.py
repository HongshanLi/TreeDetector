import json

annotation='/home/hongshan/data/annotations/instances_train2017.json'
def open_ann(ann_file):
    with open(annotation) as f:
        z=json.load(f)
    return z

def input_img_hist(ann_file):
    '''Draw histo gram of input img
    This will help us with cropping and 
    padding imgs
    '''
    z = open_ann(ann_file)
    width = []
    height = []
    for img in z['images']:
        width.append(img['width'])
        height.append(img['height'])

    return max(width), max(height)


data_dir='/home/hongshan/data/Trees'

import os
for x in os.listdir(data_dir):
    print(len(os.listdir(os.path.join(data_dir, x))))



