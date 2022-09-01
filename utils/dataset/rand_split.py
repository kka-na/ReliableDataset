import glob
import random
import os
import shutil
import sys
from pathlib import Path

if len(sys.argv) != 2 :
        print("[Usage]: python3 rand_split.py iter#")
        sys.exit()

iter_num = int(sys.argv[1])

# Get all paths to your images files and text files
base = "/home/kana/Documents/Dataset/TS/2DOD/organized/"
img = sorted(glob.glob(base+"data/*.jpg"))
label = sorted(glob.glob(base+"label/*.txt"))

# Calculate number of files for training, test
data_size = len(img)
sep_size = int(data_size*0.33)

# Shuffle two list
data_set = list(zip(img, label))
random.seed(1)
random.shuffle(data_set)
img, label = zip(*data_set)

# Now split them
a_img = img[:sep_size]
a_label = label[:sep_size]
b_img = img[sep_size:2*sep_size]
b_label = label[sep_size:2*sep_size]
c_img = img[2*sep_size:]
c_label = label[2*sep_size:]

# Move them to train, valid folders
a_folder = base+'cleaning/iter{}/a/'.format(iter_num)
b_folder = base+'cleaning/iter{}/b/'.format(iter_num)
c_folder = base+'cleaning/iter{}/c/'.format(iter_num)

os.makedirs(a_folder, exist_ok=True)
os.makedirs(b_folder, exist_ok=True)
os.makedirs(c_folder, exist_ok=True)

os.makedirs(base+'cleaning/iter{}/a/data/'.format(iter_num), exist_ok=True)
os.makedirs(base+'cleaning/iter{}/b/data/'.format(iter_num), exist_ok=True)
os.makedirs(base+'cleaning/iter{}/c/data/'.format(iter_num), exist_ok=True)

def move(paths, folder):
    for p in paths:
        shutil.copy(p, folder)


move(a_img, a_folder+"data/")
move(a_label, a_folder+"data/")
move(b_img, b_folder+"data/")
move(b_label, b_folder+"data/")
move(c_img, c_folder+"data/")
move(c_label, c_folder+"data/")