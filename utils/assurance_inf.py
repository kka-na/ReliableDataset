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

if len(sys.argv) != 4 :
    print("[Usage]: python3 inference.py dataset_name at ass")
    sys.exit()

dataset_name = str(sys.argv[1])
at = str(sys.argv[2])
ass = str(sys.argv[3])

val_path = f"{base_path}/{dataset_name}/assurance/{at}_val.txt"
inference_val_path =  f"{base_path}/{dataset_name}/assurance/{at}_by_{ass}/"
os.makedirs(inference_val_path, exist_ok=True)

def init_cfg():               
    cfg = get_cfg()
    cfg.merge_from_file(f"../training/configs/{dataset_name}/assurance_{at}_{ass}.yaml")
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.MODEL.WEIGHTS = f"../training/output/{dataset_name}/assurance_{at}_{ass}/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def inference():
    cfg = init_cfg()
    predictor = DefaultPredictor(cfg)

    with open(val_path, 'r+') as ff:
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
            txt_path = inference_val_path + os.path.splitext(os.path.basename(line))[0]+'.txt'
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
print("Valsets Inferenced [ ", dataset_name, at, ass, " ] Successfully.")
sys.exit(0)      
