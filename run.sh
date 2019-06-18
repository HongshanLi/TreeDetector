#!/bin/bash
python main.py --data=/mnt/efs/Trees_processed/ -j=2 --lr=0.0001 -b=16 \
        --print-freq=10 --train --start-epoch=1 --resume=model_10.pth \
        --epochs=20 --ckp-dir=ckps

# stop the instance after training
aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4
