#!/usr/bin/env bash
#chmod +x dataset_convert.sh
#./dataset_convert.sh iter_num

DO=/home/kana/Documents/Training/Reliable_Dataset/utils/Yolo-to-COCO-format-converter/main.py

#Convert YOLO to COCO
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/a/train/data/ --output /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/a/train/a_train.json
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/a/val/data --output /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/a/val/a_val.json
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/b/train/data --output /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/b/train/b_train.json
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/b/val/data --output /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/b/val/b_val.json
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/c/train/data --output  /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/c/train/c_train.json
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/c/val/data --output /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/c/val/c_val.json
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/eval/train/data --output /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/eval/train/eval_train.json
python $DO --path /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/eval/val/data --output /home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter$1/eval/val/eval_val.json