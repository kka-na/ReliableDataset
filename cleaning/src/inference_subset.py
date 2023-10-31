import detectron2
import sys
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import cv2
import os
import sys
from tqdm import tqdm 

base_path = "/home/kana/Documents/Dataset"

if len(sys.argv) != 5 :
    print("[Usage]: python3 hyp_tuning.py datset_name  iter# modeltset# subset#")
    sys.exit()

dataset_name = str(sys.argv[1])
iter_num = int(sys.argv[2])
modelset = str(sys.argv[3])
subset = str(sys.argv[4])

train_path = base_path+"/{}/deleting/iter{}/{}_train.txt".format(dataset_name, iter_num, modelset)
inference_path = base_path+"/{}/deleting/iter{}/{}/inference_{}/".format(dataset_name, iter_num, modelset, subset)
os.makedirs(inference_path, exist_ok=True)


def init_cfg():               
    cfg = get_cfg()
    cfg.merge_from_file("../output/{}/iter{}/{}/config.yaml".format(dataset_name, iter_num, subset))
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "../output/{}/iter{}/{}/model_final.pth".format(dataset_name, iter_num, subset)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def inference():
    cfg = init_cfg()
    predictor = DefaultPredictor(cfg)
    f = open(train_path, 'r')
    img_list = f.readlines()
    img_list = list(map(lambda s:s.strip(), img_list))
    for img in tqdm(img_list):
        im = cv2.imread(img, 1)
        outputs = predictor(im)
        h,w,c = im.shape
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
                    # img_w = str(w)+' '
                    # img_h = str(h)+'\n'
                pred_list = [pred_cls, pred_score, pred_cx, pred_cy, pred_w, pred_h]#, img_w, img_h]
                f.writelines(pred_list)

inference()
print("Subsets Inferenced [ ", dataset_name, iter_num, modelset, subset, " ] Successfully.")
sys.exit(0)    
