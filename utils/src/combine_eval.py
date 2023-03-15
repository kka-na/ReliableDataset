import glob
import random
import os
import shutil
import sys
from pathlib import Path

if len(sys.argv) != 2 :
        print("[Usage]: python3 combine_eval.py iter#")
        sys.exit()

iter_num = int(sys.argv[1])

# Get all paths to your images files and text files
base = "/home/kana/Documents/Dataset/TS/2DOD/organized/"
img = sorted(glob.glob(base+"data/*.jpg"))
label = sorted(glob.glob(base+"label/*.txt"))

# Move them to train, valid folders
eval_folder = base+'cleaning/iter{}/eval/'.format(iter_num)

os.makedirs(eval_folder, exist_ok=True)

os.makedirs(base+'cleaning/iter{}/eval/data/'.format(iter_num), exist_ok=True)

def move(paths, folder):
    for p in paths:
        shutil.copy(p, folder)


move(img, eval_folder+"data/")
move(label, eval_folder+"data/")