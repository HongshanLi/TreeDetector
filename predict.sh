#!/bin/bash

dirname=$(pwd)
python $dirname/src/main.py --predict --root=$dirname --model-ckp=$dirname/ckps/model_10.pth \
        --images=/mnt/efs/tmp_images --mask-dir=/mnt/efs/tmp_masks


