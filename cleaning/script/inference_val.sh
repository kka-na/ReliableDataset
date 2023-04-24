#!/usr/bin/env bash
#chmod +x inference_val.sh
#./inference_val.sh iter_num model_set

CUDA_VISIBLE_DEVICES=0 python ../src/inference_val.py $1 $2 a &
CUDA_VISIBLE_DEVICES=1 python ../src/inference_val.py $1 $2 b &
CUDA_VISIBLE_DEVICES=2 python ../src/inference_val.py $1 $2 c