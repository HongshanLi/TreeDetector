#!/bin/bash
ROOT=$(pwd)

python src/main.py  --train --model=resnet \
        --root=$ROOT --learning-rate=0.0001 --batch-size=32 \
        --log-dir=$ROOT/logs --ckp-dir=$ROOT/ckps \
        --resume=model_15.pth \
        --print-freq=10 --start-epoch=15 --epochs=1 \

# stop the instance after training

echo do you want to stop the instance?
read -t 10 varname
if [ $varname=='yes' -o $varname=="" ]; then
        echo shutting down the instance
        #aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4

fi

