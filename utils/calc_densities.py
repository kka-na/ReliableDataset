import os
import sys
import glob
import io
from imgsize import get_size
from tqdm import tqdm
import utils.calc_module as calc_module
from PyQt5.QtCore import *


class CalcDensities(QObject):
    def __init__(self, info):
        super(CalcDensities, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data/"
        self.whitening_path = f"{self.base_path}/whitening/"
        dir_list = os.listdir(self.whitening_path)
        calc_density_try_count = len([dir for dir in dir_list if os.path.isdir(os.path.join(self.whitening_path, dir))])
        self.target_density = (9-calc_density_try_count)*10

        with open(f"{self.base_path}/classes.txt", "r") as f:
            self.classes = [line.rstrip() for line in f.readlines()]

    send_success = pyqtSignal()
    def calc_densities(self):
        if self.target_density <= 0:
            return
        last_data_list = f"{self.iter_path}/data_deleted_with_score.txt"
        density_list = []#["Whitening-Indicators path final_score class_density object_size_density deleting_score\n"]
        with open (last_data_list, 'r') as f:
            lines = f.readlines()
            self.set_variance_list(lines)
            for line in lines:
                file_name, deleting_score = line.split(' ')
                txt_file = f"{self.data_path}/{file_name}.txt"
                width, height = self.get_image_size(file_name)
                class_density = self.calc_class_densities(txt_file)
                object_size_density = self.calc_object_size_densities(txt_file, width, height)
                final_score = float(deleting_score)*((0.5*class_density)+(0.5*object_size_density))
                density_list.append((file_name, final_score, class_density, object_size_density, float(deleting_score)))
        sorted_density_list = sorted(density_list[1:],key=lambda x: x[1], reverse=True)
        target_data_cnt = int(len(sorted_density_list)*self.target_density)
        sorted_density_list = sorted_density_list[:target_data_cnt]
        density_result_txt = f"{self.whitening_path}/try{self.target_density}.txt"
        with open(density_result_txt, 'w') as f:
            f.write("path final_score class_density object_size_density deleting_score\n")
            for sdl in sorted_density_list:
                sdl_str = "\t".join(str(x) for x in sdl)
                f.write(sdl_str+"\n")
        
        self.send_success.emit()
    
    def set_variance_list(self, lines):
        #TODO: Get Class Each Noramalized Std Deviation number's list
        class_num = len(self.classes) #Count Factor
        all_gt_bbox_count = 0  #Normalize Factor
        all_gt_bbox_class = [0]*class_num # counting whole gt's bounding box's each class number

        #TODO: Get Objec Size Noramlized Std Deviation number's list
        size_num = 5
        all_gt_bbox_size = [0]*5
        
        for line in lines: 
            txt_file = f"{self.data_path}/{line.split(' ')[0]}.txt"
            all_gt_bbox_count += calc_module.get_bbox_cnt(txt_file)
            calc_module.get_bbox_class_cnt_list(txt_file, all_gt_bbox_class)
            width, height = self.get_image_size(line.split(' ')[0])
            calc_module.get_bbox_size_cnt_list(txt_file, all_gt_bbox_size, width, height)

        class_std_var, class_dev_list = calc_module.calc_norm_variance(all_gt_bbox_class, class_num, all_gt_bbox_count)
        self.class_z_scores = calc_module.calc_z_score(class_num, class_std_var, class_dev_list)
        size_std_var, size_dev_list = calc_module.calc_norm_variance(all_gt_bbox_size, size_num, all_gt_bbox_count)
        self.size_z_scores = calc_module.calc_z_score(size_num, size_std_var, size_dev_list)

    def calc_class_densities(self, file):
        density = 0.0
        class_list = calc_module.get_bbox_class_list(file)
        for cls in class_list:
            density += float(self.class_z_scores[int(cls)])
        density /= float(len(class_list))
        return density

    def calc_object_size_densities(self, file, width, height):
        density = 0.0
        size_list = calc_module.get_bbox_size_list(file, width, height)
        for size in size_list:
            density += float(self.size_z_scores[size])
        density /= float(len(size_list))
        return density
    
    def get_image_size(self, name):
        img_file = f"{self.data_path}/{name}.jpg"
        if not os.path.exists(img_file):
            img_file = f"{self.data_path}/{name}.png"
        with io.open(img_file, 'rb') as fobj:
            width, height = get_size(fobj)
        return width, height