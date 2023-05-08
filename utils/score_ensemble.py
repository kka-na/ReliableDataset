import os
import sys
import glob
from tqdm import tqdm
import utils.calc_module as calc_module
import cv2
from datetime import datetime
from utils.inference import Inference
from PyQt5.QtCore import *


class ScoreEnsemble(QObject):
    def __init__(self, info):
        super(ScoreEnsemble, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.inference = Inference(info)
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data"
        self.sub_list = ['a','b','c']
        self.models = {"a": ("b", "c"), "b": ("a", "c"), "c": ("a", "b")}

    send_deleted = pyqtSignal(str, int)
    send_success = pyqtSignal()
    def score_ensemble(self):
        self.inference.inference()
        score_thresholds = self.calc_score_threshold()
        #score_thresholds = [0.5439, 0.5409, 0.5819]
        ensembled_score_thresholds = self.calc_ensembled_score_threshold(score_thresholds)
        whole_deleted_number = 0
        for i, _sub in enumerate(self.sub_list):
            whole_deleted_number += self.make_list(_sub, ensembled_score_thresholds[i])
        self.send_deleted.emit('eval', whole_deleted_number)
        self.send_success.emit()

    send_score = pyqtSignal(str, float)
    def calc_score_threshold(self):
        score_threshold_list = []
        def _name(path):
            return os.path.splitext(os.path.basename(path))[0]
        for _sub in self.sub_list:
            frame_cnt = 0
            score_sum = 0.0
            inf_list =  glob.glob(f"{self.iter_path}/{_sub}/val_inference/*.txt")
            for inf_path in tqdm(inf_list) :
                if _name(inf_path) == "log":
                    continue
                gt_path = f"{self.data_path}/{_name(inf_path)}.txt"
                img_path = f"{self.data_path}/{_name(inf_path)}.jpg"
                if not os.path.exists(img_path):
                    img_path =  f"{self.data_path}/{_name(inf_path)}.png"
                im = cv2.imread(img_path, 1)
                h,w,c = im.shape
                if os.path.getsize(inf_path) != 0 and os.path.getsize(gt_path) != 0 :
                    score_sum += calc_module.calc_score_threshold(gt_path, inf_path, w, h, im)
                    frame_cnt += 1

            average_score = score_sum / ( frame_cnt + 1e-10 )
            self.make_log(_sub, average_score)
            self.send_score.emit(_sub, average_score)
            score_threshold_list.append(average_score)
        return score_threshold_list


    def make_log(self, _sub, average_score):
        log_path = f"{self.iter_path}/{_sub}/log.txt"
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S ')
        if os.path.isfile(log_path): 
            f = open(log_path, 'a')
            f.write("\n" + now_time + str(average_score))
            f.close()
        else:
            f = open(log_path, 'w')
            f.write("whole_files average_score\n")
            f.write(now_time + str(average_score))
            f.close()

    def calc_means(self, gt_path, inf_path):
        gt_bboxes = calc_module.get_label_list(0, gt_path)
        net_bboxes = calc_module.get_label_list(1, inf_path)
        mconf = calc_module.calc_mconf(net_bboxes)
        ious, _, _ = calc_module.calc_ious(0.3, gt_bboxes, net_bboxes)
        miou = calc_module.calc_mious(ious)
        return mconf, miou

    def calc_ensembled_score_threshold(self, score_thresholds):
        ensembled_score_threshold_list = []
        for i, _sub in enumerate(self.sub_list):
            th_sum = 0
            for j, _th in enumerate(score_thresholds):
                if j!= i:
                    th_sum += _th
            ensembled_score_threshold = float(th_sum)/(2.0+10e-10)
            ensembled_score_threshold_list.append(ensembled_score_threshold)
        return ensembled_score_threshold_list

    
    def make_list(self, _sub, ensembled_score_threshold):
        m1, m2 = self.models[_sub]
        inf1_dir = f"{self.iter_path}/{_sub}/inference_{m1}"
        inf2_dir = f"{self.iter_path}/{_sub}/inference_{m2}"

        cleaning_list = []
        labels_cnt = 0
        cleaning_cnt = 0

        score_result_txt = f"{self.iter_path}/{_sub}/filtering_score_list.txt"
        f = open(score_result_txt, 'w')
        f.write("Filtering Score Results "+str(labels_cnt)+"\n")

        inf_dirs = [inf1_dir, inf2_dir]
        for i, inf_dir in enumerate(inf_dirs):
            if len(glob.glob(f"{inf_dir}/*.txt")) <= len(glob.glob(f"{inf_dirs[1-i]}/*.txt")):
                labels_cnt = len(glob.glob(f"{inf_dir}/*.txt"))
                for inf_path in tqdm(sorted(glob.glob(f"{inf_dir}/*.txt"))):
                    file_name = str(os.path.splitext(os.path.basename(inf_path))[0])
                    gt_path = f"{self.data_path}/{file_name}.txt"
                    inf2_path = f"{inf_dirs[1-i]}/{file_name}.txt"
                    img_path = f"{self.data_path}/{file_name}.jpg"
                    if not os.path.exists(img_path):
                        img_path = f"{self.data_path}/{file_name}.png"
                    im = cv2.imread(img_path, 1)
                    h, w, c = im.shape
                    score = self.calc_score(gt_path, inf_path, inf2_path, w, h)
                    f.write(file_name + " " + str(score) + "\n")
                    if score <= ensembled_score_threshold:
                        cleaning_list.append(file_name)
                        cleaning_cnt += 1
        f.close()

        cleaning_txt = f"{self.iter_path}/{_sub}/deleting_list.txt"
        f = open(cleaning_txt, 'w')
        f.write("Cleaning Results "+str(labels_cnt)+" -> "+str(cleaning_cnt)+"\n")
        for file_name in cleaning_list:
            f.write(file_name+"\n")
        f.close()
        self.send_deleted.emit(_sub, cleaning_cnt)
        return cleaning_cnt
    

    def calc_score(self, gt_path, inf1_path, inf2_path, width, height):
        score = 0.0

        if os.path.getsize(gt_path) !=0 and os.path.getsize(inf1_path) != 0 and os.path.getsize(inf2_path) != 0 :
            inf1_score = float(calc_module.calc_score(gt_path, inf1_path, width, height))
            inf2_score = float(calc_module.calc_score(gt_path, inf2_path, width, height))
            score = float(inf1_score+inf2_score)/(2.0+10e-10)
        return score