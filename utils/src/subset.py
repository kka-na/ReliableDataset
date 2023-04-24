import random
from tqdm import tqdm
import sys

if len(sys.argv) != 3 :
    print("[Usage]: python3 rand_split.py dataset_name iter#")
    sys.exit()

dataset_name = str(sys.argv[1])
iter_num = int(sys.argv[2])

base = "/home/kana/Documents/Dataset/{}/".format(dataset_name)
iter_folder = f"{base}cleaning/iter{iter_num}/"
data_list_txt = f"{iter_folder}data.txt"
with open(data_list_txt) as f:
    data = [line.strip() for line in f]

random.seed(1)
shuffled = random.sample(data, len(data))
separation_size = len(data) // 3

split_data = {'a': shuffled[:separation_size],
              'b': shuffled[separation_size:2*separation_size],
              'c': shuffled[2*separation_size:],
              'eval': shuffled}

def write_txt(a, data_list):
    with open(f"{iter_folder}{a}.txt", 'w') as f:
        f.writelines(f"{file}\n" for file in data_list)

def split(a, data_list):
    random.seed(1)
    shuffled = random.sample(data_list, len(data_list))
    split_size = int(len(shuffled) * 0.8)
    train_data, val_data = shuffled[:split_size], shuffled[split_size:]

    write_txt(f"{a}_train", train_data)
    write_txt(f"{a}_val", val_data)

for a, data_list in tqdm(split_data.items()):
    write_txt(a, data_list)
    split(a, data_list)