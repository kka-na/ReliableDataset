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
        self.iter_path = f"{self.base_path}/deleting/iter{self.iter}"
        self.data_path = f"{self.base_path}/data"
        self.sub_list = ['a','b','c']
        self.models = {"a": ("b", "c"), "b": ("a", "c"), "c": ("a", "b")}

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
            if len(lines) < 11:
                return
            randomlist = random.sample(range(1, len(lines)-1), 10)
            for i in randomlist:
                sample_img = self.get_random_results(lines[i].strip(), _sub)
                random_samples.append(sample_img)
        self.send_random_samples.emit(random_samples)
        self.send_success.emit()

    
    def get_random_results(self, file_name, _sub):
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
        net1, net2 = self.models[_sub]
        net1_label = f"{self.iter_path}/{_sub}/inference_{net1}/{file_name}.txt"
        net1 = calc_module.get_net_bbox(net1_label, w, h)
        net1_image = calc_module.get_result(image, net1, self.classes)
        net2_label = f"{self.iter_path}/{_sub}/inference_{net2}/{file_name}.txt"
        net2 = calc_module.get_net_bbox(net2_label, w, h)
        net2_image = calc_module.get_result(image, net2, self.classes)
        return_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        save_path = f"{self.sample_path}/{file_name}.{ext}"
        concat_img = cv2.hconcat([gt_image, net1_image, net2_image])
        #concat_img = cv2.resize(concat_img, (int(concat_img.shape[1]/2), int(concat_img.shape[0]/2)),interpolation = cv2.INTER_AREA)
        cv2.imwrite(save_path, concat_img)
        return return_image