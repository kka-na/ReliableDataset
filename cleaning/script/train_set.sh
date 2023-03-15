#!/usr/bin/env bash
#chmod +x train_set.sh
#./train_set.sh data_base_path iter_num

python ../src/train_set.py $1 $2 a &
python ../src/train_set.py $1 $2 b &
python ../src/train_set.py $1 $2 c &
python ../src/train_set.py $1 $2 eval 