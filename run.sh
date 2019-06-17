#!/bin/bash
python main.py /mnt/efs/babyTrees_processed/ -j=2 --lr=0.0001 -b=2 \
        --print-freq=1 --train --epochs=2
