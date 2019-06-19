#!/bin/bash
python main.py --data=/mnt/efs/Trees_processed/ -j=2 --lr=0.0001 -b=16 \
        --print-freq=10 --train --start-epoch=1 --epochs=10 --ckp-dir=elv_ckps/ \
        --log-dir=elv_logs/

# stop the instance after training
echo do you want to stop the instance?
read -t 10 varname
if [ $varname=no ]; then
        echo keep the instance running
        exit 1
fi

aws ec2 stop-instances --instance-ids i-03a10ed43404c35c4
