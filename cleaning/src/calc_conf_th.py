import os
import sys
import glob
from tqdm import tqdm
import calc_module
import cv2
from datetime import datetime

class CalcConfTh():
    def __init__(self):
        super(CalcConfTh, self).__init__()
        self.iter = 0
        self.subset = "a"
        self.iou_th = 0.3 # iter0 : 0.3
        self.width = 0
        self.height = 0
        self.base_path = "/home/kana/Documents/Dataset"
        self.dataset_name = "TS"

    def calc_threshold(self):
        self.inference_path = "{}/{}/deleting/iter{}/{}/val_inference/".format(self.base_path, self.dataset_name, self.iter, self.subset)

        mconf_sum = 0.0
        miou_sum = 0.0
        frame_cnt = 0
        score_sum = 0.0
        
        gt_dir = self.base_path+"/{}/data/".format(self.dataset_name)
        inf_list =  glob.glob(self.inference_path+"*.txt")
        for inf_path in tqdm(inf_list) :
            if os.path.splitext(os.path.basename(inf_path))[0] == 'log_iter{}'.format(self.iter):
                continue
            gt_path = gt_dir+os.path.splitext(os.path.basename(inf_path))[0]+'.txt'
            img_path =  gt_dir+os.path.splitext(os.path.basename(inf_path))[0]+'.jpg'
            if not os.path.exists(img_path):
                img_path =  gt_dir+os.path.splitext(os.path.basename(inf_path))[0]+'.png'
            im = cv2.imread(img_path, 1)
            h,w,c = im.shape
            if os.path.getsize(inf_path) != 0 and os.path.getsize(gt_path) != 0 :
                # First Experiments, calculate avg_conf
                # mconf, miou = self.calc_means(gt_path, inf_path)
                # self.score_instances(gt_path, inf_path)
                # miou_sum += miou
                # if miou >= self.iou_th:
                #     mconf_sum += mconf
                #     frame_cnt += 1

                # Second Experiments, calculate e*conf + iou\
                score_sum += calc_module.calc_score_threshold(gt_path, inf_path, w, h, im)
                frame_cnt += 1

        #First Experiments
        # average_mconf = mconf_sum / ( frame_cnt + 1e-10 )
        # average_miou = miou_sum / len(inf_list)

        # Second Experiments
        average_score = score_sum / ( frame_cnt + 1e-10 )

        log_path = "{}/{}/deleting/iter{}/{}/log_iter{}.txt".format(self.base_path, self.dataset_name, self.iter, self.subset, self.iter)
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S ')
        if os.path.isfile(log_path): 
            f = open(log_path, 'a')
            #f.write("\n" + now_time + str(average_mconf) + " " + str(average_miou))
            f.write("\n" + now_time + str(average_score))
            f.close()
        else:
            f = open(log_path, 'w')
            # First Experiments
            # f.write("whole_files average_mean_confidence average_mean_iou\n")
            # f.write(now_time + str(average_mconf) + " " + str(average_miou))

            # Second Experiments
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
        
if __name__ == "__main__":
    if len(sys.argv) != 4 :
        print("[Usage]: python3 calc_conf_th.py dataset_name iter# subset#")
        sys.exit()
    cct = CalcConfTh()
    cct.dataset_name = str(sys.argv[1])
    cct.iter = int(sys.argv[2])
    cct.subset = str(sys.argv[3])
    cct.calc_threshold()
    print("Threshold Calculated [ ", str(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), " ] Successfully.")
    sys.exit(0)

