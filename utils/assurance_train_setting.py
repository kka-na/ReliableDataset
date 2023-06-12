from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg

import yaml
import os
import json
import shutil
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

class TrainSetting():
    def __init__(self, info):
        super(TrainSetting, self).__init__()
        self.dataset_name = info[0]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        with open(f"{self.base_path}/classes.txt", "r") as f:
            classes = [line.rstrip() for line in f.readlines()]
        self.class_num = len(classes)
        self.assurance_path = f"{self.base_path}/assurance"
        self.data_path = f"{self.base_path}/data"
        self.assurance_list = ['a', 'b']
        self.model_list = ['faster_rcnn_R_101_FPN_3x.yaml', 'faster_rcnn_X_101_32x8d_FPN_3x.yaml']

    def train_setting(self):
        for _at in ["before", "after"]:
            train_data = f"assurance_{_at}_train"
            val_data = f"assurance_{_at}_val"
            register_coco_instances(train_data, {}, f"{self.assurance_path}/{_at}_train.json", self.data_path)
            register_coco_instances(val_data, {}, f"{self.assurance_path}/{_at}_val.json", self.data_path)

            for i, _ass in enumerate(self.assurance_list):
                cfg_file = self.get_cfg_file(self.model_list[i], train_data, val_data, _at, _ass)
                with open(f"./training/configs/{self.dataset_name}/assurance_{_at}_{_ass}.yaml", 'w') as f:
                    yaml.dump(cfg_file, f)

    def get_cfg_file(self, model_name, train_data, val_data, at, ass):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}"))
        cfg.DATASETS.TRAIN=(train_data,)
        cfg.DATASETS.TEST=(val_data,)
        cfg.TEST.EVAL_PERIOD = 500
        cfg.DATALOADER.NUM_WORKERS = 9
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}")
        cfg.SOLVER.IMS_PER_BATCH = 9
        cfg.SOLVER.BASE_LR = 0.01
        cfg.SOLVER.MAX_ITER = 5000
        cfg.SOLVER.STEPS = []
        cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.class_num 
        cfg.OUTPUT_DIR = f"./training/output/{self.dataset_name}/assurance_{at}_{ass}/"
        cfg_file = yaml.safe_load(cfg.dump())
        return cfg_file