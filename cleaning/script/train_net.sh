#!/usr/bin/env bash
#chmod +x train_net.sh
#./train_net.sh base_path iter_num

CUDA_VISIBLE_DEVICES=0,1 python ../src/train_net.py --num-gpus 2 --config-file ../configs/$1/iter$2/$3.yaml 