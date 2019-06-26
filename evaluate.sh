#!/bin/bash
dirname=$(pwd)

python $dirname/src/main.py --evaluate --root=$dirname \
        --model=unet \
        --model-ckp=$dirname/unet_ckps/model_15.pth \
        --threshold=0.5 --batch-size=16
