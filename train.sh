#!/bin/bash
ROOT=$(pwd)

python src/main.py  --train \
        --root=$ROOT --learning-rate=0.0001 --batch-size=1 \
        --log-dir=$ROOT/unet_log --ckp-dir=$ROOT/unet_ckps \
        --print-freq=1 --epochs=10 --resume=model_1.pth \
        --start-epoch=2

# stop the instance after training

echo do you want to stop the instance?
read -t 10 varname
if [ $varname=='yes' -o $varname=="" ]; then
        echo shutting down the instance

fi

#aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4
