import glob
import os
import shutil
import sys
from pathlib import Path

base = "/home/kana/Documents/Dataset/TS/2DOD/organized/"
data = base+"deleting/iter1/eval/data/"

img = sorted(glob.glob(data+"*.jpg"))
label = sorted(glob.glob(data+"*.txt"))


data_folder = base+'data/'
label_folder = base+'label/'

os.mkdir(data_folder)
os.mkdir(label_folder)

def move(paths, folder):
    for p in paths:
        shutil.copy(p, folder)
    
move(img, data_folder)
move(label, label_folder)