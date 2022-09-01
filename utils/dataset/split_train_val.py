import glob
import random
import os
import shutil
import sys
from pathlib import Path

if len(sys.argv) != 3 :
        print("[Usage]: python3 combine_eval.py iter# set_name")
        sys.exit()

iter_num = int(sys.argv[1])
set_name = str(sys.argv[2])

# Get all paths to your images files and text files
base = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter{}/{}/".format(iter_num, set_name)
img = sorted(glob.glob(base+"data/*.jpg"))
label = sorted(glob.glob(base+"data/*.txt"))

# Calculate number of files for training, test
data_size = len(img)
train_size = int(data_size*0.8)

print(data_size, len(label))

# Shuffle two list
data_set = list(zip(img, label))
random.seed(1)
random.shuffle(data_set)
img, label = zip(*data_set)

# Now split them
train_img = img[:train_size]
train_label = label[:train_size]
val_img = img[train_size:]
val_label = label[train_size:]


# Move them to train, valid folders
train_folder = base+'train/'.format(iter_num, set_name)
val_folder = base+'val/'.format(iter_num, set_name)

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

os.makedirs(base+'train/data/'.format(iter_num, set_name), exist_ok=True)
os.makedirs(base+'val/data/'.format(iter_num, set_name), exist_ok=True)

def move(paths, folder):
    for p in paths:
        shutil.copy(p, folder)


move(train_img, train_folder+"data/")
move(train_label, train_folder+"data/")
move(val_img, val_folder+"data/")
move(val_label, val_folder+"data/")