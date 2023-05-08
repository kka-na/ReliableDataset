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
        self.last_reduct = info[2]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        with open(f"{self.base_path}/classes.txt", "r") as f:
            classes = [line.rstrip() for line in f.readlines()]
        self.class_num = len(classes)
        whitening_path = f"{self.base_path}/whitening"
        self.reduct = self.get_best_reduct()
        self.reduct_path = f"{whitening_path}/reduct{self.reduct}"
        self.data_path = f"{self.base_path}/data/"
        self.assurance_list = ['a', 'b']
        self.type_list = ['train', 'val']
        self.model_list = ['rpn_R_50_C4_1x.yaml', 'rpn_R_50_FPN_1x.yaml']

    def get_best_reduct(self):
        path = f"./log/{self.dataset_name}_whitening.json"
        if os.path.exists(path):
            reduct_list = [100 - 10*n for n in range(1, (100-self.last_reduct)//10 + 1)]
            maximum_ap = 0
            best_reduct = 90
            with open(path, 'r') as f:
                data = json.load(f)
                for reduct in reduct_list:
                    if f"Reduct{reduct}" in data:
                        ap = data[f"Reduct{reduct}"]["Accuracy"]
                        if ap >= maximum_ap:
                            maximum_ap = ap
                            best_reduct = reduct
        else:
            best_reduct = 90
        return best_reduct

    def train_setting(self):
        for i, _ass in enumerate(self.assurance_list):
            train_data = f"assurance_{_ass}_train"
            val_data = f"assurance_{_ass}_val"
            register_coco_instances(train_data, {}, f"{self.reduct_path}/train.json", self.data_path)
            register_coco_instances(val_data, {}, f"{self.reduct_path}/val.json", self.data_path)
            
            cfg_file = self.get_cfg_file(self.model_list[i], train_data, val_data, _ass)

            with open(f"./training/configs/{self.dataset_name}/assurance_{_ass}.yaml", 'w') as f:
                yaml.dump(cfg_file, f)

    def get_cfg_file(self, model_name, train_data, val_data, ass):
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
        cfg.OUTPUT_DIR = f"./training/output/{self.dataset_name}/assurance_{ass}/"
        cfg_file = yaml.safe_load(cfg.dump())
        return cfg_file