#!/bin/bash

python src/main.py --evaluate  \
        --model=resnet \
        --model-ckp=ckps/model_10.pth \
        --threshold=0.5 --batch-size=64
