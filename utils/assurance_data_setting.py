from pathlib import Path
import shutil
import json
import os
from PyQt5.QtCore import pyqtSignal, QObject

class DataSetting(QObject):
    def __init__(self, info):
        super(DataSetting, self).__init__()
        self.dataset_name = info[0]
        self.last_reduct = info[2]
        self.init_path()
    
    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter1"
        whitening_path = f"{self.base_path}/whitening"
        self.reduct = self.get_best_reduct()
        self.reduct_path = f"{whitening_path}/reduct{self.reduct}"
        self.assurance_path = f"{self.base_path}/assurance"
        if not os.path.exists(self.assurance_path):
            os.mkdir(self.assurance_path)
        self.data_path = f"{self.base_path}/data"

    
    def get_best_reduct(self):
        path = f"./log/{self.dataset_name}_whitening.json"
        if os.path.exists(path):
            reduct_list = [100 - 10*n for n in range(1, (100-self.last_reduct)//10 + 1)]
            maximum_ap = 0
            best_reduct = 90
            with open(path, 'r') as f:
                data = json.load(f)
                for reduct in reduct_list:
                    if f"Reduct{reduct}" in data:
                        ap = data[f"Reduct{reduct}"]["Accuracy"]
                        if ap >= maximum_ap:
                            maximum_ap = ap
                            best_reduct = reduct
        else:
            best_reduct = 90
        return best_reduct
    
    send_best_reduct = pyqtSignal(int)
    def data_setting(self):
        data_before_txt = f"{self.iter_path}/data.txt"
        shutil.copy(data_before_txt, f"{self.assurance_path}/data_before.txt")
        data_after_txt = f"{self.reduct_path}/data.txt"
        shutil.copy(data_after_txt, f"{self.assurance_path}/data_after.txt")

        data_before_train = f"{self.iter_path}/eval_train.json"
        data_before_val = f"{self.iter_path}/eval_val.json"
        shutil.copy(data_before_train, f"{self.assurance_path}/before_train.json")
        shutil.copy(data_before_val, f"{self.assurance_path}/before_val.json")

        data_after_train = f"{self.reduct_path}/train.json"
        data_after_val = f"{self.reduct_path}/val.json"
        shutil.copy(data_after_train, f"{self.assurance_path}/after_train.json")
        shutil.copy(data_after_val, f"{self.assurance_path}/after_val.json")
        self.send_best_reduct.emit(self.reduct)
