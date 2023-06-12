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
        self.iter = info[1]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        with open(f"{self.base_path}/classes.txt", "r") as f:
            classes = [line.rstrip() for line in f.readlines()]
        self.class_num = len(classes)
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data/"
        self.sub_list = ['a','b','c','eval']
        self.type_list = ['train', 'val']
        self.model_list = ['faster_rcnn_R_101_C4_3x.yaml', 'faster_rcnn_R_101_DC5_3x.yaml','faster_rcnn_R_101_FPN_3x.yaml','rpn_R_50_FPN_1x.yaml']

    def train_setting(self):
        def check_and_make(path):
            if os.path.exists(path):
                shutil.rmtree(path)  
            os.mkdir(path)

        check_and_make(f"./training/configs/{self.dataset_name}/iter{self.iter}/")
        check_and_make(f"./training/output/{self.dataset_name}/iter{self.iter}/")
        
        for i,_sub in enumerate(self.sub_list):
            train_data = f"iter{self.iter}_{_sub}_train"
            val_data = f"iter{self.iter}_{_sub}_val"
            register_coco_instances(train_data, {}, f"{self.iter_path}/{_sub}_train.json", self.data_path)
            register_coco_instances(val_data, {}, f"{self.iter_path}/{_sub}_val.json", self.data_path)
            
            cfg_file = self.get_cfg_file(self.model_list[i], train_data, val_data, _sub)

            with open(f"./training/configs/{self.dataset_name}/iter{self.iter}/{_sub}.yaml", 'w') as f:
                yaml.dump(cfg_file, f)

    def get_cfg_file(self, model_name, train_data, val_data, sub):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}"))
        cfg.DATASETS.TRAIN=(train_data,)
        cfg.DATASETS.TEST=(val_data,)
        cfg.TEST.EVAL_PERIOD = 500
        cfg.DATALOADER.NUM_WORKERS = 12
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}")
        cfg.SOLVER.IMS_PER_BATCH = 12
        cfg.SOLVER.BASE_LR = 0.01
        cfg.SOLVER.MAX_ITER = 5000
        cfg.SOLVER.STEPS = []
        cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.class_num 
        cfg.OUTPUT_DIR = f"./training/output/{self.dataset_name}/iter{self.iter}/{sub}"
        cfg_file = yaml.safe_load(cfg.dump())
        return cfg_file