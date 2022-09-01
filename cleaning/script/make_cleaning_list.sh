#!/usr/bin/env bash
#chmod +x make_cleaning_list.sh
#./make_cleaning_list.sh iter_num

python ../src/make_cleaning_list.py $1 a &
python ../src/make_cleaning_list.py $1 b &
python ../src/make_cleaning_list.py $1 c
