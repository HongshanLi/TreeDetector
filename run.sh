#!/bin/bash
python main.py /mnt/efs/Trees_processed/ -j=2 --lr=0.0001 -b=16 \
        --print-freq=10 --train --epochs=2
