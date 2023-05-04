import cv2
import os
import utils.calc_module as calc_module
import random
from PyQt5.QtCore import *

class CheckDeleting(QObject):
    def __init__(self, info):
        super(CheckDeleting, self).__init__ ()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.init_path()

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data"
        self.sub_list = ['a','b','c']

        self.sample_path = f"{self.iter_path}/deleted_sample/"
        os.makedirs(self.sample_path, exist_ok=True)

        with open(f"{self.base_path}/classes.txt", "r") as f:
            self.classes = [line.rstrip() for line in f.readlines()]       

    send_random_samples = pyqtSignal(object)
    send_success = pyqtSignal()
    def check_deleting(self):
        random_samples = []
        for _sub in self.sub_list:
            deleting_list = f"{self.iter_path}/{_sub}/deleting_list.txt"
            f = open(deleting_list, 'r')
            lines = f.readlines()
            if len(lines) < 4:
                return
            randomlist = random.sample(range(1, len(lines)-1), 3)
            for i in randomlist:
                sample_img = self.get_random_results(lines[i].strip())
                random_samples.append(sample_img)
        self.send_random_samples.emit(random_samples)
        self.send_success.emit()

    
    def get_random_results(self, file_name):
        ext = "jpg"
        img_path = f"{self.data_path}/{file_name}.{ext}"
        if not os.path.exists(img_path):
            ext = "png"
            img_path = f"{self.data_path}/{file_name}.{ext}"
        image = cv2.imread(img_path, 1)
        h,w,c = image.shape
        gt_label = f"{self.data_path}/{file_name}.txt"
        gt = calc_module.get_gt_bbox(gt_label, w, h)
        gt_image = calc_module.get_result(image, gt, self.classes)
        save_path = f"{self.sample_path}/{file_name}.{ext}"
        cv2.imwrite(save_path, gt_image)
        return_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        return return_image