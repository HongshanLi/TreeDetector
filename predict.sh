#!/bin/bash

python src/main.py --predict --model=unet \
        --model-ckp=unet_ckps/model_10.pth \
        --images=images --mask-dir=masks


