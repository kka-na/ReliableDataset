from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg

import yaml
import os
import shutil
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

class TrainSetting():
    def __init__(self, info):
        super(TrainSetting, self).__init__()
        self.dataset_name = info[0]
        self.reduct = info[2]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        with open(f"{self.base_path}/classes.txt", "r") as f:
            classes = [line.rstrip() for line in f.readlines()]
        self.class_num = len(classes)
        whitening_path = f"{self.base_path}/whitening"
        self.reduct_path = f"{whitening_path}/reduct{self.reduct}"
        self.data_path = f"{self.base_path}/data/"
        self.type_list = ['train', 'val']
        self.model = 'rpn_R_50_FPN_1x.yaml'
    

    def train_setting(self):
        train_data = f"reduct{self.reduct}_train"
        val_data = f"reduct{self.reduct}_val"
        register_coco_instances(train_data, {}, f"{self.reduct_path}/train.json", self.data_path)
        register_coco_instances(val_data, {}, f"{self.reduct_path}/val.json", self.data_path)
        
        cfg_file = self.get_cfg_file(self.model, train_data, val_data)

        with open(f"./training/configs/{self.dataset_name}/reduct{self.reduct}.yaml", 'w') as f:
            yaml.dump(cfg_file, f)

    def get_cfg_file(self, model_name, train_data, val_data):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}"))
        cfg.DATASETS.TRAIN=(train_data,)
        cfg.DATASETS.TEST=(val_data,)
        cfg.TEST.EVAL_PERIOD = 500
        cfg.DATALOADER.NUM_WORKERS = 8
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.1
        cfg.SOLVER.MAX_ITER = 20#5000
        cfg.SOLVER.STEPS = []
        cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.class_num 
        cfg.OUTPUT_DIR = f"./training/output/{self.dataset_name}/reduct{self.reduct}"
        cfg_file = yaml.safe_load(cfg.dump())
        return cfg_file