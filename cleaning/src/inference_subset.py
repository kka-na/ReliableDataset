import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import yaml
import glob
import cv2
import os
import sys

base_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/"

if len(sys.argv) != 4 :
    print("[Usage]: python3 hyp_tuning.py iter# modeltset# subset#")
    sys.exit()

iter_num = int(sys.argv[1])
modelset = str(sys.argv[2])
subset = str(sys.argv[3])

train_path = base_path+"iter{}/{}/train/data/".format(iter_num, subset)
inference_path = base_path+"iter{}/{}/train/inference_{}/".format(iter_num, subset, modelset)
os.makedirs(inference_path, exist_ok=True)

w = 1920
h = 1080 

def init_cfg():               
    cfg = get_cfg()
    cfg.merge_from_file("../output/iter{}_{}_train/config.yaml".format(iter_num, modelset))
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "../output/iter{}_{}_train/model_0004999.pth".format(iter_num, modelset)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def inference():
    cfg = init_cfg()
    predictor = DefaultPredictor(cfg)
    for img in glob.glob(train_path+"*.jpg"):
        im = cv2.imread(img)
        outputs = predictor(im)
        txt_path = inference_path + os.path.splitext(os.path.basename(img))[0]+'.txt'
        with open(txt_path, 'w') as f :
            num = int(outputs["instances"].pred_classes.size(0))
            for i in range(num):
                pred_cls = str(int(outputs["instances"].pred_classes[i]))+' '
                pred_score = str(float(outputs["instances"].scores[i]))+' '
                pred_cx, pred_cy, pred_w, pred_h = '','','',''
                for bbox in outputs["instances"].pred_boxes[i]:
                    box_w = float(bbox[2]-bbox[0])
                    box_h = float(bbox[3]-bbox[1])
                    pred_cx = str(float(bbox[0]+(box_w/2))/w)+' '
                    pred_cy = str(float(bbox[1]+(box_h/2))/h)+' '
                    pred_w = str(box_w/w)+' '
                    pred_h = str(box_h/h)+'\n'
                pred_list = [pred_cls, pred_score, pred_cx, pred_cy, pred_w, pred_h]
                f.writelines(pred_list)

inference()
    
