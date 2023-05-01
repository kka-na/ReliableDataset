import sys
import os
from tqdm import tqdm
from PyQt5.QtCore import *

class Cleaning(QObject):
    def __init__(self, info):
        super(Cleaning, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.init_path()

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data/"
        self.sub_list = ['a','b','c']

    send_success = pyqtSignal()
    send_ratio = pyqtSignal(float)
    def cleaning(self):
        f_pre_list = []
        with open(f"{self.iter_path}/data.txt", 'r') as f_pre:
            f_pre_list = set(line.strip() for line in f_pre)
        before_len = len(f_pre_list)

        for _sub in self.sub_list:
            file = f"{self.iter_path}/{_sub}/deleting_list.txt"
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    name = line.strip()
                    img = f"{self.data_path}/{name}.jpg"
                    if not os.path.exists(img):
                        img = f"{self.data_path}/{name}.png"
                    f_pre_list.discard(img)
        
        with open(f"{self.iter_path}/data_deleted.txt", 'w') as f_aft:
            for n in f_pre_list:
                f_aft.write("%s\n"%n)
        after_len = len(f_pre_list)
        deleting_ratio = float(after_len/before_len)
        self.send_success.emit()
        self.send_ratio.emit(deleting_ratio)