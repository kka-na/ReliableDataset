#!/usr/bin/env bash
#chmod +x dataset_setting.sh
#./dataset_setting.sh iter_num

python ../dataset/rand_split.py $1
python ../dataset/combine_eval.py $1
python ../dataset/split_train_val.py $1 a &
python ../dataset/split_train_val.py $1 b &
python ../dataset/split_train_val.py $1 c &
python ../dataset/split_train_val.py $1 eval