#!/usr/bin/env bash
#chmod +x calc_conf_th.sh
#./calc_conf_th.sh iter_num

python ../src/calc_conf_th.py $1 a &
python ../src/calc_conf_th.py $1 b &
python ../src/calc_conf_th.py $1 c