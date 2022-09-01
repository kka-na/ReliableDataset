import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import sys
import yaml

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

if len(sys.argv) != 7 :
    print("[Usage]: python3 calc_conf_th.py iter# subset# batch_size_per_image img_per_batch iteration learning_rate")
    sys.exit()

iter_num = int(sys.argv[1])
subset = str(sys.argv[2])
batch_size_per_image = int(sys.argv[3])
ims_per_batch = int(sys.argv[4])
iteration = int(sys.argv[5])
learning_rate = float(sys.argv[6])

base_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/"
train_path = base_path+"iter{}/{}/train/".format(iter_num, subset)
val_path = base_path+"iter{}/{}/val/".format(iter_num, subset)
train_data = "iter{}_{}_train".format(iter_num, subset)
val_data = "iter{}_{}_val".format(iter_num, subset)

register_coco_instances(train_data, {}, train_path+"{}_train.json".format(subset), train_path+"data")
register_coco_instances(val_data, {}, val_path+"{}_val.json".format(subset), val_path+"data")

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

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.DATASETS.TRAIN=(train_data,)
cfg.DATASETS.TEST=(val_data,)
cfg.TEST.EVAL_PERIOD = 500
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
cfg.SOLVER.BASE_LR = learning_rate
cfg.SOLVER.MAX_ITER = iteration
cfg.SOLVER.STEPS = []
cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 
cfg.OUTPUT_DIR = "../output/"+train_data

cfg_file = yaml.safe_load(cfg.dump())
with open("../configs/iter{}_{}.yaml".format(iter_num, subset), 'w') as f:
    yaml.dump(cfg_file, f)
