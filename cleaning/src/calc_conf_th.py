import os
import sys
import glob
import calc_module
from datetime import datetime

class CalcConfTh():
    def __init__(self):
        super(CalcConfTh, self).__init__()
        self.iter = 0
        self.subset = "a"
        self.iou_th = 0.3 # iter0 : 0.3
        self.width = 1920
        self.height = 1080

    def calc_threshold(self):
        self.gt_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter1/{}/val/".format(self.subset)
        self.inference_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter{}/{}/val/".format(self.iter, self.subset)

        mconf_sum = 0.0
        miou_sum = 0.0
        frame_cnt = 0
        score_sum = 0.0
        
        gt_dir = self.gt_path+"data/"
        inf_list =  glob.glob(self.inference_path+"inference/*.txt")
        for inf_path in inf_list :
            gt_path = gt_dir+os.path.splitext(os.path.basename(inf_path))[0]+'.txt'
            if os.path.getsize(inf_path) != 0 and os.path.getsize(gt_path) != 0 :
                # First Experiments, calculate avg_conf
                # mconf, miou = self.calc_means(gt_path, inf_path)
                # self.score_instances(gt_path, inf_path)
                # miou_sum += miou
                # if miou >= self.iou_th:
                #     mconf_sum += mconf
                #     frame_cnt += 1

                # Second Experiments, calculate e*conf + iou
                # confs, ious, classes = self.score_instances(gt_path, inf_path)
                # conf = 0
                # correct_cnt = 0
                # score = 0 
                # #print(ious)
                # for i in range(len(confs)):
                #     conf = classes[i] if confs[i] else 0
                #     score += conf + float(list(ious[i].values())[0])
                #     correct_cnt += 1
                score_sum += calc_module.calc_each_score(gt_path, inf_path)
                frame_cnt += 1

        #First Experiments
        # average_mconf = mconf_sum / ( frame_cnt + 1e-10 )
        # average_miou = miou_sum / len(inf_list)

        # Second Experiments
        average_score = score_sum / ( frame_cnt + 1e-10 )

        log_path = self.inference_path + "log_iter{}.txt".format(self.iter)
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
        ious, _, _ = calc_module.calc_ious(gt_bboxes, net_bboxes)
        miou = calc_module.calc_mious(ious)
        return mconf, miou

    def score_instances(self, gt_path, inf_path):
        gt_bboxes = calc_module.get_label_list(0, gt_path)
        net_bboxes = calc_module.get_label_list(1, inf_path)
        ious, checked_gt, checked_net = calc_module.calc_ious(gt_bboxes, net_bboxes)
        confs = calc_module.get_confs(net_bboxes, checked_net)
        classes = calc_module.get_classes(gt_bboxes, checked_gt, ious)
        return confs, ious, classes

if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("[Usage]: python3 calc_conf_th.py iter# subset#")
        sys.exit()
    cct = CalcConfTh()
    cct.iter = int(sys.argv[1])
    cct.subset = str(sys.argv[2])
    cct.calc_threshold()

