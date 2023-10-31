import os
import io
import shutil
from imgsize import get_size
import calc_module as calc_module
from PyQt5.QtCore import *

class CheckDensities(QObject):
    def __init__(self, info):
        super(CheckDensities, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/deleting/iter{self.iter}"
        self.data_path = f"{self.base_path}/data"
        self.whitening_path = f"{self.base_path}/whitening"
        
        with open(f"{self.base_path}/classes.txt", "r") as f:
            self.classes = [line.rstrip() for line in f.readlines()]
        print(self.classes)


    def check_densities(self):

        last_data_list = f"{self.iter_path}/data_deleted_with_score.txt"
        with open (last_data_list, 'r') as f:
            lines = f.readlines()
            class100, bbox100 = self.set_distribution_list(lines)
        with open (f"{self.whitening_path}/reduct90/data.txt", 'r') as f:
            lines = f.readlines()
            class90, bbox90 = self.get_distribution_list(lines)
        
        with open (f"{self.whitening_path}/reduct80/data.txt", 'r') as f:
            lines = f.readlines()
            class80, bbox80 = self.get_distribution_list(lines)
        
        with open (f"{self.whitening_path}/reduct70/data.txt", 'r') as f:
            lines = f.readlines()
            class70, bbox70 = self.get_distribution_list(lines)
        
        with open (f"{self.whitening_path}/reduct60/data.txt", 'r') as f:
            lines = f.readlines()
            class60, bbox60 = self.get_distribution_list(lines)
        
        with open (f"{self.whitening_path}/reduct50/data.txt", 'r') as f:
            lines = f.readlines()
            class50, bbox50 = self.get_distribution_list(lines)
           
        result_txt = f"{self.whitening_path}/distribution_result.txt"
        with open(result_txt, 'w') as f:
            f.write(' '.join(self.classes))
            f.write('\n')
            f.write(' '.join(str(x) for x in self.bbox_size_categories))
            f.write('\n')
            f.write(' '.join(str(x) for x in class100))
            f.write('\n')
            f.write(' '.join(str(x) for x in bbox100))
            f.write('\n')
            f.write(' '.join(str(x) for x in class90))
            f.write('\n')
            f.write(' '.join(str(x) for x in bbox90))
            f.write('\n')
            f.write(' '.join(str(x) for x in class80))
            f.write('\n')
            f.write(' '.join(str(x) for x in bbox80))
            f.write('\n')
            f.write(' '.join(str(x) for x in class70))
            f.write('\n')
            f.write(' '.join(str(x) for x in bbox70))
            f.write('\n')
            f.write(' '.join(str(x) for x in class60))
            f.write('\n')
            f.write(' '.join(str(x) for x in bbox60))
            f.write('\n')
            f.write(' '.join(str(x) for x in class50))
            f.write('\n')
            f.write(' '.join(str(x) for x in bbox50))
            f.write('\n')

    def set_distribution_list(self, lines):
        class_num = len(self.classes) 
        all_gt_bbox_count = 0  
        all_gt_bbox_class =  [ 0 for _ in range(class_num)]  
        size_num = 5
        all_gt_bbox_size = [ 0 for _ in range(size_num)]
        min_bbox_size = 999999
        max_bbox_size = 0
        for line in lines: 
            txt_file = f"{self.data_path}/{line.split(' ')[0]}.txt"
            all_gt_bbox_count += calc_module.get_bbox_cnt(txt_file)
            calc_module.get_bbox_class_cnt_list(txt_file, all_gt_bbox_class)
            width, height = self.get_image_size(line.split(' ')[0])
            bbox_sizes = calc_module.get_bbox_size(txt_file, width, height)
            for bs in bbox_sizes:
                if bs < min_bbox_size:
                    min_bbox_size = bs
                if bs > max_bbox_size:
                    max_bbox_size = bs
        self.bbox_size_categories = calc_module.get_bbox_size_categories(min_bbox_size, max_bbox_size, size_num)
        for line in lines: 
            txt_file = f"{self.data_path}/{line.split(' ')[0]}.txt"
            width, height = self.get_image_size(line.split(' ')[0])
            calc_module.get_bbox_size_cnt_list(txt_file, all_gt_bbox_size, width, height, self.bbox_size_categories)

        return all_gt_bbox_class, all_gt_bbox_size

    def get_distribution_list(self, lines):
        class_num = len(self.classes) 
        all_gt_bbox_class =  [ 0 for _ in range(class_num)]  
        all_gt_bbox_size = [ 0 for _ in range(len(self.bbox_size_categories)-1)]

        for line in lines: 
            txt_file = f"{line.split('.')[0]}.txt"
            calc_module.get_bbox_class_cnt_list(txt_file, all_gt_bbox_class)
            width, height = self.get_image_size((line.split('/')[-1]).split('.')[0])
            calc_module.get_bbox_size_cnt_list(txt_file, all_gt_bbox_size, width, height, self.bbox_size_categories)

        return all_gt_bbox_class, all_gt_bbox_size

    def get_image_size(self, name):

        img_file = f"{self.data_path}/{name}.jpg"
        if not os.path.exists(img_file):
            img_file = f"{self.data_path}/{name}.png"
        try:
            with io.open(img_file, 'rb') as fobj:
                width, height = get_size(fobj)
        except:
            width = 1920
            height = 1080
        return width, height

if __name__ == "__main__":
    cd = CheckDensities(('AIHub2', '4'))
    cd.check_densities()
