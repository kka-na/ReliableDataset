import os
import sys
import glob
import calc_module
from datetime import datetime
from tqdm import tqdm

class MakeCleaningList():
    def __init__(self):
        super(MakeCleaningList, self).__init__()
        self.iter = 0
        self.subset = "a"
        self.target1 = "b"
        self.target2 = "c"

        self.score_th = 0.8
        self.width = 1920
        self.height = 1080

    def set_values(self):
        if self.subset == "a":
            self.target1 = "b"
            self.target2 = "c"
        elif self.subset == "b":
            self.target1 = "a"
            self.target2 = "c"
        elif self.subset == "c":
            self.target1 = "a"
            self.target2 = "b"
        
        self.base_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter{}/".format(self.iter)
        self.data_train_path = self.base_path+"{}/train/".format(self.subset)
        self.data1_val_path = self.base_path+"{}/val/".format(self.target1)
        self.data2_val_path = self.base_path+"{}/val/".format(self.target2)
        
        score_th1 = self.get_score(self.data1_val_path)
        score_th2 = self.get_score(self.data2_val_path)
        self.score_th = ( score_th1 + score_th2 )/ 2

    
    def get_score(self, path):
        log_path = path + "log_iter{}.txt".format(self.iter)
        f = open(log_path, 'r')
        lines = f.readlines()
        for line in lines:
            pass
        score = float(line.split()[2])
        return score

    def make_list(self):
        self.set_values()

        gt_dir = self.data_train_path+"data/"
        inf1_dir =  self.data_train_path+"inference_{}/".format(self.target1)
        inf2_dir =  self.data_train_path+"inference_{}/".format(self.target2)

        cleaning_list = []
        labels_cnt = 0
        cleaning_cnt = 0

        
        if len(glob.glob(inf1_dir+"*.txt")) <= len(glob.glob(inf2_dir+"*.txt")):
            labels_cnt = len(glob.glob(inf1_dir+"*.txt"))
            for inf1_path in sorted(glob.glob(inf1_dir+"*.txt")):
                gt_path = gt_dir+os.path.splitext(os.path.basename(inf1_path))[0]+'.txt'
                inf2_path = inf2_dir+os.path.splitext(os.path.basename(inf1_path))[0]+'.txt'
                score = self.calc_score(gt_path, inf1_path, inf2_path)
                if score <= self.score_th:
                    cleaning_list.append(str(os.path.splitext(os.path.basename(inf1_path))[0]))
                    cleaning_cnt += 1

        elif len(glob.glob(inf2_dir+"*.txt")) < len(glob.glob(inf1_dir+"*.txt")):
            labels_cnt = len(glob.glob(inf2_dir+"*.txt"))
            for inf2_path in sorted(glob.glob(inf2_dir+"*.txt")):
                gt_path = gt_dir+os.path.splitext(os.path.basename(inf1_path))[0]+'.txt'
                inf1_path = inf1_dir+os.path.splitext(os.path.basename(inf2_path))[0]+'.txt'
                score = self.calc_score(gt_path, inf1_path, inf2_path)
                if score <= self.score_th:
                    cleaning_list.append(str(os.path.splitext(os.path.basename(inf2_path))[0]))
                    cleaning_cnt += 1

        cleaning_txt = self.data_train_path+"cleaning_list.txt"
        f = open(cleaning_txt, 'w')
        f.write("Cleaning Results "+str(labels_cnt)+" -> "+str(cleaning_cnt)+"\n")
        for file_name in cleaning_list:
            f.write(file_name+"\n")
        f.close()
    
    def calc_score(self, gt_path, inf1_path, inf2_path):
        score = 0.0
        if os.path.getsize(gt_path) !=0 and os.path.getsize(inf1_path) != 0 and os.path.getsize(inf2_path) != 0 :
            # inf1_bboxes = calc_module.get_label_list(1, inf1_path)
            # inf2_bboxes = calc_module.get_label_list(1, inf2_path)
            # mconf1 = calc_module.calc_mconf(inf1_bboxes)
            # mconf2 = calc_module.calc_mconf(inf2_bboxes)
            # score = (mconf1+mconf2)/2

            inf1_score = calc_module.calc_each_score(gt_path, inf1_path)
            inf2_score = calc_module.calc_each_score(gt_path, inf2_path)

            score = (inf1_score+inf2_score)/2
        return score
    
   
if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("[Usage]: python3 calc_conf_th.py iter# subset#")
        sys.exit()
    cct = MakeCleaningList()
    cct.iter = int(sys.argv[1])
    cct.subset = str(sys.argv[2])
    cct.make_list()

