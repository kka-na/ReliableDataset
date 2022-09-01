#!/usr/bin/env bash
#chmod +x inference_subset.sh
#./inference_subset.sh iter_num model_set sub_set 

SUBSET1="b"
SUBSET2="c"

if [[ "$2" == "a" ]]
then
    SUBSET1="b"
    SUBSET2="c"
elif [[ "$2" == "b" ]]
then
    SUBSET1="a"
    SUBSET2="c"
elif [[ "$2" == "c" ]]
then
    SUBSET1="a"
    SUBSET2="b"
fi

CUDA_VISIBLE_DEVICES=0 python ../src/inference_subset.py $1 $2 $SUBSET1 &
CUDA_VISIBLE_DEVICES=1 python ../src/inference_subset.py $1 $2 $SUBSET2