#!/usr/bin/env bash
#chmod +x calc_conf_th.sh
#./calc_conf_th.shd  datset_name iter_num

python ../src/calc_conf_th.py $1 $2 a &
python ../src/calc_conf_th.py $1 $2 b &
python ../src/calc_conf_th.py $1 $2 c