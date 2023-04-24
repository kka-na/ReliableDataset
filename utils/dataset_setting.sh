#!/usr/bin/env bash
#chmod +x dataset_setting.sh
#./dataset_setting.sh dataset_path iter_num 

ROUTE=/home/kana/Documents/Dataset/$1/cleaning/iter$2

python ./src/subset.py $1 $2
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/a_train.txt --output  $ROUTE/a_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/a_val.txt --output  $ROUTE/a_val.json
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/b_train.txt --output  $ROUTE/b_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/b_val.txt --output  $ROUTE/b_val.json
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/c_train.txt --output  $ROUTE/c_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/c_val.txt --output  $ROUTE/c_val.json
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/eval_train.txt --output  $ROUTE/eval_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $ROUTE/eval_val.txt --output  $ROUTE/eval_val.json