#!/bin/bash
dirname=$(pwd)

python $dirname/src/main.py --evaluate --data=/mnt/efs/Trees_processed \
        --model-ckp=$dirname/ckps/model_3.pth \
        --threshold=0.5 --batch-size=32
