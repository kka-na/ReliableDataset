import cv2
import os
import utils.calc_module as calc_module
import random
from PyQt5.QtCore import *

class CheckCleaning(QObject):
    def __init__(self, info):
        super(CheckCleaning, self).__init__ ()
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

    send_random_samples = pyqtSignal(object)
    send_success = pyqtSignal()
    def check_cleaning(self):
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
        gt_image = self.get_result(image, gt)
        save_path = f"{self.sample_path}/{file_name}.{ext}"
        cv2.imwrite(save_path, gt_image)
        return_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        return return_image
    
    def get_result(self, image,bboxes):
        image_cp = image.copy()
        for i, di in enumerate(bboxes):
            ll = list(di.items())[0]
            color = calc_module.get_color(int(ll[0]))
            name = str(i) + " " +calc_module.get_name(int(ll[0]))
            cv2.putText(image_cp, name, (int(ll[1][0]),int(ll[1][1])-2),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA )
            cv2.rectangle(image_cp, (int(ll[1][0]),int(ll[1][1])), (int(ll[1][2]),int(ll[1][3])),color, 3)
        return image_cp