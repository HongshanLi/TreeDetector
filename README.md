# TreeDetector

## Data
The raw data I had from Aerialytic consists of 1250 x 1250 aerial RGBA images, 
point-cloud LiDAR images (for elevation), infrared pixelpeak images,
and hand-labellel mask for trees (as RGB images). 
They are proprietary to Aerialytic, so I cannot release them. 

In my finalized model, I only used the RGB channels of the 
aerial RGBA images and their corresponding masks. 

### preprocess
Pre-process consists of:
- Divide 1250x1250 RGBA images into 25 250x250 sub-images
- Remove the A channel from the images
- Divide 1250x1250 masks into 25 250x250 sub-masks
- Compute mean and standard deviation of the input images (channel-wise, as float32) 
I divided the images into subimages because it is too big for the model.
More precisely, the GPU I was using is unable to hold the model when 
it is doing inference (with gradient tape) on one image.

Moreover, if I can create submasks on subimages, then I only need to piece
together the submasks to get the mask for the whole images.

To preprocess the data, create a file `raw_image_mask.csv` in the project
root directory. The `raw_image_mask.csv` file needs to consists of two 
columns. The first column consists the full path to each RGBA images
and the second column consists of full path of the corresponding masks.

Then run
```
bash preprocess.sh
```
After the process is finished, you should see `proc_data/` in the project
root directory. `proc_data/` consists of two sub-direcoties 
`proc_data/imgs/` and `proc_data/masks/`, and a json file `mean_std.json`.
The sub-directories constains the 250x250 RGB images 
and their corresponding masks, respectively. Name of the images match 
the name of the mask. For example, the mask of `00000.png` in `proc_data/imgs/`
will be `00000.png` in `proc_data/masks/`. `mean_std.json` file contains the
mean and standard deviation of the images.

Of course, you can create your own script to preprocess the data, as long as
you put the processed images and masks in `proc_data/imgs/` 
and `proc_data/masks/` and you create a json file containing 
```
{
"mean" :[<R-channel mean>, <G-channel mean>, <B-channel mean>],
"std": [<R-channel std>, <G-channel std>, <B-channel std>]
}
```

## Train
Once you have the preprocessed data ready in `proc_data/`, to train the model,
run
```
python src/main.py --train
```
You can configure the training process by adding more flags
```
--epochs
--start-epoch
--resume
--ckp-dir
--log-dir
```
checkpoints of the model after each epoch will be saved at `--ckp-dir`. 
Model trained after epoch N will be saved as `model_N.pth`. 

Log of the whole training process will be saved at `--log-dir` as 
`log.pickle`.


## Evaluate
To find the checkpoint with the best validation accuracy, do
```
python src/main.py --find-best-model
```
If you changed `--log-dir` flag in training, you need to specify it here as well
```
python src/main.py --find-best-model --log-dir=[dir to the log]
```

To evaluate the model performance on test set, do
```
python src/main.py --evaluate --model-ckp=[path to the model checkpoint]
```









