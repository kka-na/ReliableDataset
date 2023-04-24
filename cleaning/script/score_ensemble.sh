#!/usr/bin/env bash
#./inference_val.sh iter_num model_set

CUDA_VISIBLE_DEVICES=0 python ../src/inference_val.py $1 $2 a &
CUDA_VISIBLE_DEVICES=1 python ../src/inference_val.py $1 $2 b &
CUDA_VISIBLE_DEVICES=2 python ../src/inference_val.py $1 $2 c


CUDA_VISIBLE_DEVICES=0 python ../src/inference_subset.py $1 $2 a b &
CUDA_VISIBLE_DEVICES=1 python ../src/inference_subset.py $1 $2 a c &
CUDA_VISIBLE_DEVICES=2 python ../src/inference_subset.py $1 $2 b a 

CUDA_VISIBLE_DEVICES=0 python ../src/inference_subset.py $1 $2 b c &
CUDA_VISIBLE_DEVICES=1 python ../src/inference_subset.py $1 $2 c a &
CUDA_VISIBLE_DEVICES=2 python ../src/inference_subset.py $1 $2 c b

exit 0
