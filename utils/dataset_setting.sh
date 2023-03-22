#!/usr/bin/env bash
#chmod +x dataset_setting.sh
#./dataset_setting.sh dataset_path iter_num 

python ./src/subset.py $2
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/a_train.txt --output  $1/iter$2/a_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/a_val.txt --output  $1/iter$2/a_val.json
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/b_train.txt --output  $1/iter$2/b_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/b_val.txt --output  $1/iter$2/b_val.json
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/c_train.txt --output  $1/iter$2/c_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/c_val.txt --output  $1/iter$2/c_val.json
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/eval_train.txt --output  $1/iter$2/eval_train.json
python ./Yolo-to-COCO-format-converter/main.py --path $1/iter$2/eval_val.txt --output  $1/iter$2/eval_val.json