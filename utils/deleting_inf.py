import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm 

import cv2
import os
import sys

base_path = "/home/kana/Documents/Dataset"

if len(sys.argv) != 5 :
    print("[Usage]: python3 inference.py datset_name iter# modelset# subset#")
    sys.exit()

dataset_name = str(sys.argv[1])
iter_num = int(sys.argv[2])
modelset = str(sys.argv[3])
subset = str(sys.argv[4])
iter_path = f"{base_path}/{dataset_name}/cleaning/iter{iter_num}"

if modelset == subset:
    target_path = f"{base_path}/{dataset_name}/cleaning/iter1/{modelset}_val.txt"
    inference_path = f"{iter_path}/{modelset}/val_inference/"
else:
    target_path = f"{iter_path}/{modelset}_train.txt"
    inference_path = f"{iter_path}/{modelset}/inference_{subset}/"

os.makedirs(inference_path, exist_ok=True)


def init_cfg():               
    cfg = get_cfg()
    cfg.merge_from_file(f"../training/configs/{dataset_name}/iter{iter_num}/{subset}.yaml")
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.MODEL.WEIGHTS = f"../training/output/{dataset_name}/iter{iter_num}/{subset}/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def inference():
    cfg = init_cfg()
    predictor = DefaultPredictor(cfg)
    with open(target_path, 'r+') as ff:
        lines = ff.readlines()
        ff.seek(0)
        for line in tqdm(lines):
            line = line.strip()
            im = cv2.imread(line, 1)
            if im is not None: 
                ff.write(line + "\n") 
            else: 
                continue
            outputs = predictor(im)
            h,w,c = im.shape
            txt_path = inference_path + os.path.splitext(os.path.basename(line))[0]+'.txt'
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
        ff.truncate()

inference()
print("Valsets Inferenced [ ", dataset_name, iter_num, modelset, " ] Successfully.")
sys.exit(0)      
