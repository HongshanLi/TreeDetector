#!/bin/bash
python main.py --data=/mnt/efs/babyTrees_processed/ -j=2 --lr=0.0001 -b=16 \
        --print-freq=2 --train --start-epoch=1 --epochs=2 --ckp-dir=tmp/ \
        --log-dir=tmp/
# stop the instance after training
#aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4
