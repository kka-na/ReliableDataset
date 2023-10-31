import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision
import random
import os
import sys

from detectron2 import model_zoo
from engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import yaml

import wandb
wandb.login()

base_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/deleting/"

if len(sys.argv) != 3 :
    print("[Usage]: python3 hyp_tuning.py iter# subset#")
    sys.exit()

iter_num = int(sys.argv[1])
subset = str(sys.argv[2])



train_path = base_path+"iter{}/{}/train/".format(iter_num, subset)
val_path = base_path+"iter1/{}/val/".format(subset)
train_data = "iter{}_{}_train".format(iter_num, subset)
val_data = "iter{}_{}_val".format(iter_num, subset)

register_coco_instances(train_data, {}, train_path+"{}_train.json".format(subset), train_path+"data")
register_coco_instances(val_data, {}, val_path+"{}_val.json".format(subset), val_path+"data")
metadata = MetadataCatalog.get(train_data)
dataset_dicts = DatasetCatalog.get(train_data)

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
sweep_config = {
    'method': 'random'  
}
metric = {  # method를 bayes로 사용한다면 반드시 metric이 정의되어야 한다.
    'name': 'total_loss',  # name은 학습코드에서 log로 지정한 이름과 같아야 함 (total_loss는 detectron2/engine/train_loop.py 파일에서 run_step 함수에 log를 찍어놓음)
    'goal': 'minimize'
}
sweep_config['metric'] = metric
parameters_dict = {
    'learning_rate': {  
        'distribution': 'uniform',  #  uniform distribution 즉 균둥분포로 lr을 뽑겠다는 의미 
        'min' : 0.01,
        'max' : 0.1
    },
    'IMS_PER_BATCH': {
        'values': [8, 16, 32]
    },
    'iteration': {
        'values': [5000]
    },
    'BATCH_SIZE_PER_IMAGE' : {
        'values': [128, 256, 512]
    },       
    'model': {
        'value' : model_name
    }
}
# parameters_dict['model']['value'] = model_name
sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='noisy_label_cleaning_{}'.format(subset))


#Configuration
def init_cfg(config):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.model))
    cfg.DATASETS.TRAIN=(train_data)
    cfg.DATASETS.TEST=()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WIEGHTS = model_zoo.get_checkpoint_url(config.model)
    cfg.SOLVER.IMS_PER_BATCH = config.IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = config.learning_rate
    cfg.SOLVER.MAX_ITER = config.iteration
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 
    cfg.OUTPUT_DIR = "./training/output/iter{}_{}_tune".format(iter_num, subset)

    return cfg


#Training!``
def train(config=None):
    with wandb.init(project='noisy_label_cleaning_{}'.format(subset)) as run:
        config = wandb.config
        print(config)
        cfg = init_cfg(config)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()


wandb.agent(sweep_id, train, count=10)