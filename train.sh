#!/bin/bash
ROOT=$(pwd)

python src/main.py  --train --model=unet \
        --root=$ROOT --learning-rate=0.0001 --batch-size=16 \
        --log-dir=$ROOT/unet_logs --ckp-dir=$ROOT/unet_ckps \
        --print-freq=10 --epochs=15 \

# stop the instance after training

echo do you want to stop the instance?
read -t 10 varname
if [ $varname=='yes' -o $varname=="" ]; then
        echo shutting down the instance
        #aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4

fi

