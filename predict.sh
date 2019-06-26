#!/bin/bash

dirname=$(pwd)
python $dirname/src/main.py --predict \
        --root=$dirname --model-ckp=$dirname/unet_ckps/model_10.pth \
        --images=images --mask-dir=masks


