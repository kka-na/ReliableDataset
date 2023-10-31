import os
import io
import sys
import json
from imgsize import get_size
import calc_module as calc_module

ACHIVE1 = 42
ACHIVE2 = 43

class CalcQAI():
    def __init__(self, name):
        super(CalcQAI, self).__init__()
        self.dataset_name = name
        self.init_path()
      
    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.data_path = f"{self.base_path}/data"
        self.assurance_path = f"{self.base_path}/assurance"
        with open(f"{self.base_path}/classes.txt", "r") as f:
            self.classes = [line.rstrip() for line in f.readlines()]
      
    def calc_qai(self):
        for _at in ['before', 'after']:
            val_txt = f"{self.assurance_path}/{_at}_val.txt"
            with open(val_txt, "r") as f:
                file_paths = [line.strip() for line in f.readlines()]
            txts = [os.path.splitext(os.path.basename(os.path.expanduser(file_path)))[0] + '.txt' for file_path in file_paths]
            imgs = [os.path.basename(os.path.expanduser(file_path)) for file_path in file_paths]
            val_len = len(txts)

            class_num = len(self.classes) #Count Factor
            all_gt_bbox_count = 0  #Normalize Factor
            all_gt_bbox_class = [ 0 for _ in range(class_num)] # counting whole gt's bounding box's each class number
            size_num = 5
            min_bbox_size = 999999
            max_bbox_size = 0
            all_gt_bbox_size =[ 0 for _ in range(size_num)]


            width  = 0
            height = 0

            for t, i in zip(txts, imgs): 
                txt_file = f"{self.data_path}/{t}"
                all_gt_bbox_count += calc_module.get_bbox_cnt(txt_file)
                calc_module.get_bbox_class_cnt_list(txt_file, all_gt_bbox_class)
                try:
                    with io.open(f"{self.data_path}/{i}", 'rb') as fobj:
                        width,height = get_size(fobj)
                except:
                    pass
                bbox_sizes = calc_module.get_bbox_size(txt_file, width, height)
                for bs in bbox_sizes:
                    if bs < min_bbox_size:
                        min_bbox_size = bs
                    elif bs > max_bbox_size:
                        max_bbox_size = bs
            self.bbox_size_categories = calc_module.get_bbox_size_categories(min_bbox_size, max_bbox_size, size_num)


            for t,i in zip(txts, imgs):
                
                txt_file = f"{self.data_path}/{t}"
                try:
                    with io.open(f"{self.data_path}/{i}", 'rb') as fobj:
                        width,height = get_size(fobj)
                except:
                    pass
                calc_module.get_bbox_size_cnt_list(txt_file, all_gt_bbox_size, width, height,self.bbox_size_categories)
       
            class_var, _ = calc_module.calc_norm_variance(all_gt_bbox_class, class_num, all_gt_bbox_count)
            obj_size_var, _ = calc_module.calc_norm_variance(all_gt_bbox_size, size_num, all_gt_bbox_count)
        
            bbox_acc1, bbox_acc2 = 0,0
            all_a_conf_list = [ [0] for _ in range(class_num)]
            all_b_conf_list = [ [0] for _ in range(class_num)]
            w = 0
            h = 0
            for t,i in zip(txts, imgs):
                try:
                    with io.open(f"{self.data_path}/{i}", 'rb') as fobj:
                        w,h = get_size(fobj)
                except:
                    pass
                val_a = f"{self.assurance_path}/{_at}_by_a/{t}"
                val_b = f"{self.assurance_path}/{_at}_by_b/{t}"
                val_gt = f"{self.data_path}/{t}"

                gt_bboxes = calc_module.get_label_list(0, val_gt, w, h)
                a_bboxes = calc_module.get_label_list(1, val_a, w, h)
                b_bboxes = calc_module.get_label_list(1, val_b, w, h)

                a_ious, _, _ = calc_module.calc_ious(0.5, gt_bboxes, a_bboxes)
                b_ious, _, _ = calc_module.calc_ious(0.5, gt_bboxes, b_bboxes)
                bbox_acc1 += calc_module.calc_mious(a_ious)
                bbox_acc2 += calc_module.calc_mious(b_ious)

                calc_module.get_bbox_conf_list(a_bboxes, all_a_conf_list)
                calc_module.get_bbox_conf_list(b_bboxes, all_b_conf_list)
  
            ap1, ap2 = self.calc_aps(_at)
            achive1, achive2 = self.calc_achivement(ap1, ap2)
            avg_achivement = ((achive1+achive2)/2.0)*100
            avg_achivement = 100 if avg_achivement > 100 else avg_achivement
            avg_bbox_acc = ((float(bbox_acc1/val_len)+float(bbox_acc2/val_len))/2.0)*100
            
            obj_sim = self.calc_obj_sim(all_a_conf_list, all_b_conf_list)
            obj_sim *=100.0
            class_var *= 100.0
            obj_size_var *= 100.0
            print(avg_achivement, avg_bbox_acc, obj_sim, class_var, obj_size_var)
            qai = 0.2*(avg_achivement+avg_bbox_acc+(100-obj_sim)+(100-class_var)+(100-obj_size_var))
            
            print(qai)
    
    def calc_aps(self, at):
        with open(f"../log/{self.dataset_name}_assurance.json", 'r') as f:
            data = json.load(f)
            if at == "before":
                ap1 = data[""]["Before_AP1"]
                ap2 = data[""]["Before_AP2"]
            elif at == "after":
                ap1 = data[""]["After_AP1"]
                ap2 = data[""]["After_AP2"]
            return ap1, ap2
    
    def calc_achivement(self, ap1, ap2):
        achive1 =  float(ap1/ACHIVE1) if ap1 != 0 else 0
        achive2 =  float(ap2/ACHIVE2) if ap2 != 0 else 0
        return achive1, achive2

    def calc_obj_sim(self, a_conf_l, b_conf_l):
        sum_conf_var = 0 
        for i in range(len(self.classes)):
            a_conf_list = calc_module.get_bbox_sim_list(a_conf_l[i])
            a_conf_var, _ = calc_module.calc_norm_variance(a_conf_list, 5, len(a_conf_l[i]))
            b_conf_list = calc_module.get_bbox_sim_list(b_conf_l[i])
            b_conf_var, _ = calc_module.calc_norm_variance(b_conf_list, 5, len(b_conf_l[i]))
            sum_conf_var += ((a_conf_var+b_conf_var)/2.0)
        sim = float(sum_conf_var/len(self.classes))
        return sim
    
if __name__ == "__main__":
    cq = CalcQAI(str(sys.argv[1]))
    cq.calc_qai()