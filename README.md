# TreeDetector: Predict tree coverage via deep learning
This is the consulting project I worked on at Insight Data Science as
an Artificial Intelligence Fellow. The goal of this project is 
to segment trees from 2D RGB imagery.

![Project Description](./static/proj_dsp.png)

The project demo can be found [here](https://docs.google.com/presentation/d/1hNJnrgQvOk3Bi-aoHRNYCFyrtNb9LjB2eyo4sXsL4n8/edit#slide=id.g5cf1a3734f_0_6)
The deep learning models in this project are 
developed in Pytorch 1.1.0

## Setup
```
git clone https://github.com/HongshanLi/TreeDetector
cd TreeDetector
pip install -r requirements.txt
```

## Data
The relavant raw data I used are provided by the company providing
this consulting project.
It consists of 1250 x 1250 RGBA aerial imagery, 
point-cloud LiDAR imagery,
and their corresponding masks.
They are proprietary to the consulting company, so I cannot release them. 



### preprocess
Pre-process consists of:
- Divide 1250x1250 RGBA images into 25 250x250 sub-images
- Remove the A channel from the images
- Divide 1250x1250 LiDAR image into 25 250x250 sub-images
- Divide 1250x1250 masks into 25 250x250 sub-masks
- Compute mean and standard deviation of the input images after divide pixel value by 255 (channel-wise, as float32) 
I divided the images into subimages because it is too big for the model.
More precisely, when the model is doing a forward pass on even
one image, it will require more GPU memory than the one I was
using (Tesla K80)

Moreover, if I can create submasks on subimages, 
then I only need to piece
together the submasks to get the mask for the whole images.

To preprocess the data, create a file `raw_data_path.csv` in the project
root directory. The `raw_data_path.csv` file needs to consists of three
columns. Put full path of the RGBA image in the first column, full path 
of LiDAR image in the second column, and full path of mask 
in the third column.

The existing `raw_data_path.csv` 
in the repo should be a good example. It contains full paths of
samples images in `sample_images/`.


Then run
```
python src/main.py --preprocess
```
Then you should see `proc_data/` in the project repo.


## Pipeline
Only RGB images will be used for the pipeline in master branch.
Another pipeline that incorperates LiDAR images in `use_lidar`
branch.


## Models
CNN is used to extract features from image. For this project, I have 
two models to create masks, one uses ResNet152 as a backbone feature
extractor, the other one is a U-net.


## Train
Training process uses Adam optimizer

Once you have the preprocessed data ready in `proc_data/`, 
to train the resnet-based model
```
python src/main.py --train --model=resnet --epochs=[num of epochs to train]
```
If you want to use a pretrained ResNet on ImageNet, add `--pretrained`
flag, e.g.
```
python src/main.py --train --model=resnet --epochs=[num of epochs to train] --pretrained
```

To train unet-based model , run
```
python src/main.py --train --model=unet --epochs=[num of epochs to train]
```
`--pretrained` flag is only available for ResNet.


#### Advanced configurations
You can configure the training process by adding more flags
```
--batch-size=[int: batch size]
--resume=[bool: resume from the lastest ckp]
--learning-rate=[float: learning rate]
--print-freq=[int: num of steps to train before print out log]
```

## Evaluate
To find the checkpoint with the best validation accuracy, do
```
python src/main.py --find-best-model
```

To evaluate the model performance on test set, do
```
python src/main.py --evaluate --model=[resnet or unet] \
        --model-ckp=[path to the model checkpoint]
```
For example, if you want to evaluate the checkpoint of 
resnet model obtained after 10th epoch on test set, do
```
python src/main.py --evaluate --model=resnet \
        --model-ckp=resnet_ckps/model_10.pth
```

### Baseline
To compare to the baseline model add `--baseline` flag.
The baseline model is pixel thresholding. It picks out green pixels
in the RGB image and classify it as pixels inside trees. Obviously, 
there are a lot drawbacks with this baseline model. For example,
lawns are green, and trees in winter are typically not green.



## Inference
I will explain how to make inference using Resnet-based model.

As the model is trained on the proprietary data from the consulting 
company, I cannot publish the trained models on the full training 
set. But if you are interested in running inference without training
the models on your own dataset, you can download the trained checkpoint
of Resnet-based model on the sample data [here](s3://hongshan-public/model_10.pth)
It is the checkpoint after 10 epochs of training on sample images.

Then create a directory `resnet_ckps/` in the project directory
```
mkdir resnet_ckps
```
and move the downloaded checkpoint in `resnet_ckps/`.

If you trained the Resnet-based model on your own dataset,
the training process will automatically create `resnet_ckps/`
directory and save all checkpoints there.

To create mask on images, do
```
python src/main.py --predict --model=resnet \
        --model-ckps=[path to model ckp] \
        --image-dir=[directory of RGB imgs] \
        --mask-dir=[directory of predicted masks]
```
For examples, if you want to use Resnet-based model with checkpoint trained after 10th epoch,
the images you want to draw masks on are saved in `static/images/` and you want to 
save the predicted masks in `static/masks/`, do
```
python src/main.py --predict --model=resnet --model=resnet_ckps/model_10.pth \
        --image-dir=static/images/ --mask-dir=static/masks/
```







