import glob
import random
import os
import sys

if len(sys.argv) != 2 :
    print("[Usage]: python3 rand_split.py iter#")
    sys.exit()

iter_num = int(sys.argv[1])

base = "/home/kana/Documents/Dataset/TS/2DOD/"
data = sorted(glob.glob(base+"data/*.jpg"))

data_size = len(data)
separation_size = int(data_size*0.33)

shuffled = list(data)
random.seed(1)
random.shuffle(shuffled)

a_data = shuffled[:separation_size]
b_data = shuffled[separation_size:2*separation_size]
c_data = shuffled[2*separation_size:]

iter_folder = base+'cleaning/iter{}/'.format(iter_num)

os.makedirs(iter_folder, exist_ok=True)

def write_txt(a, data_list):
    txt_path = iter_folder+'{}.txt'.format(a)
    file = open(txt_path, 'w')
    for f in data_list:
        file.write("{}\n".format(f))

write_txt('a', a_data)
write_txt('b', b_data)
write_txt('c', c_data)
write_txt('eval', shuffled)


def split(a, data_list):
    data_size = len(data_list)
    split_size = int(data_size*0.8)
    shuffled = data_list
    random.seed(1)
    random.shuffle(shuffled)

    train_data = shuffled[:split_size]
    val_data = shuffled[split_size:]

    write_txt(a+"_train",train_data)
    write_txt(a+"_val", val_data)

split('a', a_data)
split('b', b_data)
split('c', c_data)
split('eval', shuffled)
    