import os
import glob

iter_num = 0
subset = "c"
from_subst = "b"

#path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter{}/{}/val/inference/".format(iter_num, subset)
path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter{}/{}/train/inference_{}/".format(iter_num, subset, from_subst)

w = 1920
h = 1080

for label in glob.glob(path+"*.txt"):
    fr = open(label, 'r')
    lines = fr.readlines()
    new_content = ""
    for line in lines:
        v = line.split()
        v_w = float(float(v[4])*w-float(v[2])*w)
        v_h = float(float(v[5])*h-float(v[3])*h)
        cx = float(float(v[2])*w+v_w/2)/w
        cy = float(float(v[3])*h+v_h/2)/h
        ww = v_w/w
        hh = v_h/h
        line = line.replace(v[2], str(cx))
        line = line.replace(v[3], str(cy))
        line = line.replace(v[4], str(ww))
        line = line.replace(v[5], str(hh))
        new_content += line

    
    fr.close()
    fw = open(label, 'w')
    fw.write(new_content)
    fw.close()