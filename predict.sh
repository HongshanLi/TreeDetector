#!/bin/bash

dirname=$(pwd)
python $dirname/src/main.py --predict --model-ckp=$dirname/ckps/model_10.pth \
        --images=$dirname/images --mask-dir=$dirname/masks

