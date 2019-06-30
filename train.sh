#!/bin/bash

python src/main.py  --train --model=resnet \
        --learning-rate=0.0001 --batch-size=3 \
        --print-freq=10 --epochs=2 \

# stop the instance after training

echo do you want to stop the instance?
read -t 10 varname
if [ $varname=='yes' -o $varname=="" ]; then
        echo shutting down the instance
        #aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4

fi

