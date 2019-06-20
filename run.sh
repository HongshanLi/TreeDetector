#!/bin/bash
python main.py --data=/mnt/efs/Trees_processed/ -j=2 --lr=0.0001 -b=16 \
        --print-freq=10 --train --resume=model_10.pth \
        --start-epoch=11 --epochs=1 --ckp-dir=ckps/ \
        --log-dir=logs/ 

# stop the instance after training

echo do you want to stop the instance?
read -t 10 varname
if [ $varname=='yes' -o $varname=="" ]; then
        echo shutting down the instance

fi

#aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4
