#!/usr/bin/env bash
#chmod +x cleaning.sh
#./cleaning.sh iter_num


python ../src/calc_conf_th.py $1 $2 a &
python ../src/calc_conf_th.py $1 $2 b &
python ../src/calc_conf_th.py $1 $2 c


python ../src/make_cleaning_list.py $1 $2 a &
python ../src/make_cleaning_list.py $1 $2 b &
python ../src/make_cleaning_list.py $1 $2 c


python ../src/cleaning.py $1 $2

exit 0