import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import sys
import yaml
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

if len(sys.argv) != 4 :
    print("[Usage]: python3 dataset_name  iter# subset#")
    sys.exit()


dataset_name  = str(sys.argv[1])
iter_num = int(sys.argv[2])
subset = str(sys.argv[3])
base_path = "/home/kana/Documents/Dataset/{}".format(dataset_name)
data_path = base_path+"data/"
iter_path = base_path+"deleting/iter{}/".format(iter_num)
train_data = "iter{}_{}_train".format(iter_num, subset)
val_data = "iter{}_{}_val".format(iter_num, subset)

register_coco_instances(train_data, {}, iter_path+"{}_train.json".format(subset), data_path)
register_coco_instances(val_data, {}, iter_path+"{}_val.json".format(subset), data_path)

def get_model():
    model_name = 'COCO-Detection/faster_rcnn_R_101_C4_3x.yaml'
    if subset == "a":
        model_name = 'COCO-Detection/faster_rcnn_R_101_C4_3x.yaml'
    elif subset == "b":
        model_name = 'COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml'
    elif subset == "c":
        model_name = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    elif subset == "eval":
        model_name = 'COCO-Detection/rpn_R_50_FPN_1x.yaml'
    return model_name

model_name = str(get_model())

os.makedirs("../configs/{}/iter{}/".format(dataset_name, iter_num), exist_ok=True)
os.makedirs("../output/{}/iter{}/".format(dataset_name, iter_num), exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.DATASETS.TRAIN=(train_data,)
cfg.DATASETS.TEST=(val_data,)
cfg.TEST.EVAL_PERIOD = 500
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.STEPS = []
cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 
cfg.OUTPUT_DIR = "../output/{}/iter{}/{}".format(dataset_name, iter_num, subset)
cfg_file = yaml.safe_load(cfg.dump())


with open("../configs/{}/iter{}/{}.yaml".format(dataset_name, iter_num, subset), 'w') as f:
    yaml.dump(cfg_file, f)
print("Train Set [ ", dataset_name, iter_num, subset, " ] Successfully.")
sys.exit(0)