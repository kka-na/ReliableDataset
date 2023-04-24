import os
import sys
import glob
import calc_module
from datetime import datetime
from tqdm import tqdm
import cv2

class MakeCleaningList():
    def __init__(self):
        super(MakeCleaningList, self).__init__()
        self.dataset_name = "TS"
        self.iter = 0
        self.subset = "a"
        self.target1 = "b"
        self.target2 = "c"

        self.score_th = 0.8

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
        
        self.base_path = "/home/kana/Documents/Dataset/{}".format(self.dataset_name)
        self.data_train_path = self.base_path+"/cleaning/iter{}/{}/".format(self.iter, self.subset)
        data1_path = self.base_path+"/cleaning/iter{}/{}/".format(self.iter, self.target1)
        data2_path = self.base_path+"/cleaning/iter{}/{}/".format(self.iter, self.target2)
        self.data1_val_path = self.data_train_path+"inference_{}/".format(self.target1)
        self.data2_val_path = self.data_train_path+"inference_{}/".format(self.target2)
        self.score_th = self.get_score(data1_path, data2_path)

    
    def get_score(self, path1, path2):
        log_path1 = path1 + "log_iter{}.txt".format(self.iter)
        log_path2 = path2 + "log_iter{}.txt".format(self.iter)
        f = open(log_path1, 'r')
        lines = f.readlines()
        for line in lines:
            pass
        score1 = float(line.split()[2])

        f = open(log_path2, 'r')
        lines = f.readlines()
        for line in lines:
            pass
        score2 = float(line.split()[2])

        score = float(score1+score2)/(2.0+10e-10)
        return score

    def make_list(self):
        self.set_values()

        gt_dir = self.base_path+"/data/"
        inf1_dir = self.base_path+"/cleaning/iter{}/{}/inference_{}/".format(self.iter, self.subset, self.target1)
        inf2_dir = self.base_path+"/cleaning/iter{}/{}/inference_{}/".format(self.iter, self.subset, self.target2)
        
        cleaning_list = []
        labels_cnt = 0
        cleaning_cnt = 0

        score_result_txt = self.data_train_path+"filtering_score_list.txt"
        f = open(score_result_txt, 'w')
        f.write("Filtering Score Results "+str(labels_cnt)+"\n")

        if len(glob.glob(inf1_dir+"*.txt")) <= len(glob.glob(inf2_dir+"*.txt")):
            labels_cnt = len(glob.glob(inf1_dir+"*.txt"))
            for inf1_path in tqdm(sorted(glob.glob(inf1_dir+"*.txt"))):
                file_name = str(os.path.splitext(os.path.basename(inf1_path))[0])
                gt_path = gt_dir+file_name+'.txt'
                inf2_path = inf2_dir+file_name+'.txt'
                img_path =  gt_dir+file_name+'.jpg'
                if not os.path.exists(img_path):
                    img_path = gt_dir+file_name+'.png'
                im = cv2.imread(img_path, 1)
                h,w,c = im.shape
                score = self.calc_score(gt_path, inf1_path, inf2_path, w, h)
                f.write(file_name+" "+str(score)+"\n")
                if score <= self.score_th:
                    cleaning_list.append(file_name)
                    cleaning_cnt += 1

        elif len(glob.glob(inf2_dir+"*.txt")) < len(glob.glob(inf1_dir+"*.txt")):
            labels_cnt = len(glob.glob(inf2_dir+"*.txt"))
            for inf2_path in tqdm(sorted(glob.glob(inf2_dir+"*.txt"))):
                file_name = str(os.path.splitext(os.path.basename(inf2_path))[0])
                gt_path = gt_dir+file_name+'.txt'
                inf1_path = inf1_dir+file_name+'.txt'
                img_path =  gt_dir+file_name+'.jpg'
                if not os.path.exists(img_path):
                    img_path = gt_dir+file_name+'.png'
                im = cv2.imread(img_path, 1)
                h,w,c = im.shape
                score = self.calc_score(gt_path, inf1_path, inf2_path, w, h)
                f.write(file_name+" "+str(score)+"\n")
                if score <= self.score_th:
                    cleaning_list.append(file_name)
                    cleaning_cnt += 1
        f.close()

        cleaning_txt = self.data_train_path+"cleaning_list.txt"
        f = open(cleaning_txt, 'w')
        f.write("Cleaning Results "+str(labels_cnt)+" -> "+str(cleaning_cnt)+"\n")
        for file_name in cleaning_list:
            f.write(file_name+"\n")
        f.close()
    
    def calc_score(self, gt_path, inf1_path, inf2_path, width, height):
        score = 0.0
        if os.path.getsize(gt_path) !=0 and os.path.getsize(inf1_path) != 0 and os.path.getsize(inf2_path) != 0 :
            # inf1_bboxes = calc_module.get_label_list(1, inf1_path)
            # inf2_bboxes = calc_module.get_label_list(1, inf2_path)
            # mconf1 = calc_module.calc_mconf(inf1_bboxes)
            # mconf2 = calc_module.calc_mconf(inf2_bboxes)
            # score = (mconf1+mconf2)/2

            inf1_score = float(calc_module.calc_score(gt_path, inf1_path, width, height))
            inf2_score = float(calc_module.calc_score(gt_path, inf2_path, width, height))

            score = float(inf1_score+inf2_score)/(2.0+10e-10)
        return score
    
   
if __name__ == "__main__":
    if len(sys.argv) != 4 :
        print("[Usage]: python3 calc_conf_th.py dataset_name iter# subset#")
        sys.exit()
    cct = MakeCleaningList()
    cct.dataset_name = str(sys.argv[1])
    cct.iter = int(sys.argv[2])
    cct.subset = str(sys.argv[3])
    cct.make_list()
    print("Made Cleaning List [ ", str(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), " ] Successfully.")
    sys.exit(0)

