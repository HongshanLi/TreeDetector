#!/bin/bash
python main.py --data=/mnt/efs/babyTrees_processed/ -j=2 --lr=0.0001 -b=16 \
        --print-freq=10 --train --epochs=3 --ckp-dir=./tmp/
