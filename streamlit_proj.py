'''Display images and predicted masks using streamlit'''
import torch
import torchvision.transforms as transforms

import numpy as np
import streamlit as st
import os
import argparse
import skimage.io as io
import src.utils as utils
import torch
import numpy as np
from PIL import Image
from src.models import ResNetModel
from src.post_process import CleanUp

def show_header(name, avatar_image_url, **links):
    links = ' | '.join('[%s](%s)' % (key, url) for key, url in links.items())
    st.write(
    """
    <img src="%s" style="border-radius:50%%;height:100px;vertical-align:text-bottom;padding-bottom:10px"/>
    <span style="display:inline-block;padding-left:10px;padding-bottom:20px;font-size:3rem;vertical-align:bottom">%s</span>
    
    %s
    """ % (avatar_image_url, name, links))


show_header(
    avatar_image_url="https://hongshan-public.s3-us-west-2.amazonaws.com/hongshan_headshot_icon.png",
    name="Hongshan Li",
    github='https://github.com/HongshanLi/TreeDetector',
    linkedin='https://www.linkedin.com/in/hongshanli/',
)

st.markdown("# Welcome to TreeDector")


st.write(
        "This is the Streamlit demo of the deep project I completed as an Artificial Intelligence Fellow at Insight Data Science. \
        The goal of the project is to train a deep learning model that can segment \
        trees from 2D aerial imagery. My best performing model uses ResNet152 as backbone feature extractor.\
        You can play with the model and see it in action here.")

@st.cache
def load_image(filename):
    img = io.imread("sample_raw_data/037185-0_RGB-Ir.tif")
    large_image = img[:,:,0:3]
    small_image = Image.fromarray(large_image)
    small_image.thumbnail((600, 600))
    small_image = np.array(small_image)
    return large_image, small_image


@st.cache
def init_clean_up():
    return CleanUp()

cleanup = CleanUp(threshold=0.5)

img, thumbnail = load_image("sample_raw_data/037185-0_RGB-Ir.tif")
#st.write(img.shape)
#st.write(thumbnail.shape)
#st.image(img, width=600)


st.write("The image below comes from the test set:")
st.image(thumbnail, use_column_width=True, caption="sample image from test set (not used in training)")

st.write("You can crop a 250 x 250 sub-image from it by moving the slide bar below. The x and y value from the slide bar will be the x and y offsets (the coordinates of the top-left corner) of the sub-image:")

x = st.slider('X offset', 0, 0, 1000, 1)
y = st.slider('Y offset', 0, 0, 1000, 1)

st.write("Once you cropped the image, the model will draw a contour (in red) around the place, where it thinks has trees.")
         

sub_img = img[y:y+250, x:x+250, :]
result_caption="Running the detector in realtime."
result = st.image(sub_img, width=250, caption=result_caption)

device = torch.device('cuda:0' if torch.cuda.is_available()
        else 'cpu')
@st.cache
def load_model():
    model = ResNetModel(pretrained=False,use_lidar=False)
    model.load_state_dict(
            torch.load('resnet_real_ckps/model_9.pth', map_location=device))


    model.to(device)
    return model

# normalize the image

model = load_model()
x = sub_img.astype(np.float32)
transform = transforms.Compose([
    transforms.ToTensor()
    ])

x = transform(x)
mean = torch.mean(x, dim=(1,2))
std = torch.std(x, dim=(1,2))

mean = mean.view(3, 1, 1)
std = std.view(3, 1, 1)

x = (x - mean) / std

x = x.unsqueeze(0)

x = x.to(device)
mask = model(x)


_,_,h,w = mask.shape

mask = mask.view(h,w)
mask = mask.detach().cpu().numpy()

mask = cleanup(mask)
mask = np.array(mask)
mask = mask[:, :, 0] / 255
mask = np.array([mask, mask, mask]).transpose((1,2,0))

sub_img = sub_img / 255
red = np.zeros(sub_img.shape)+[1,0,0]
#st.image(red)
mask = 0.5*(1 + mask)
composite = sub_img * mask + red*(1- mask)
result.image(composite, width=250, caption=result_caption)
# stack mask on top of image

#st.image([mask])
#st.button(label="test")
#x = st.slider(label="x coordinate of the crop center")
#y = st.slider(label="y coordinate of the crop center")
#st.write(x, y)




def pixelwise_accuracy(mask, target):
    '''compute pixelwise accuracy
        Args:
            mask (np.float32): black and white mask
            black = object, white = background
            target (np.float32): ...
    '''
    correct = (mask == target).astype(np.float32)

    acc = np.sum(correct) / (mask.shape[0]*mask.shape[1])
    return acc.item()

def compute_iou(mask, target):
    '''compute intersection over union
        Args:
            mask (np.float32): black and white mask
            black = object, white = background
            target (np.float32): ...
    
    '''
    # make object have pixel value 1
    mask = (mask == 0).astype(np.float32)
    target = (target == 0).astype(np.float32)

    intersection = mask*target

    union = mask + target - intersection

    iou = np.sum(intersection) / np.sum(union)
    return iou.item()




def get_background(img):
    '''
    Args:
        img (np.uint8): input image
    Return (np.float32): mask of objects in the image
    white pixel for background, black pixel for non-background
    pixel value is normalized between [0, 1]
    '''
    img = img.astype(np.float32)
    img = np.mean(img, axis=2)
    img = img / 255
    img = (img == 1).astype(np.float32)
    return img

