import os
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

from PyQt5.QtCore import pyqtSignal, QObject
import json
import wandb
import yaml
import logging

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

global trainer
def training(cfg, ass, reduct_path, data_path):
    global trainer
    # train_data = f"assurance_{ass}_train"
    # val_data = f"assurance_{ass}_val"
    # register_coco_instances(train_data, {}, f"{reduct_path}/train.json", data_path)
    # register_coco_instances(val_data, {}, f"{reduct_path}/val.json", data_path)
            
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
class TrainStart(QObject):
    def __init__(self, info):
        super(TrainStart, self).__init__()
        self.dataset_name = info[0]
        self.last_reduct = info[2]
        self.init_path() 

        logging.getLogger("detectron2").setLevel(logging.WARNING)
        logging.getLogger("detectron2").propagate = False


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
        self.gpu = 1

    def get_best_reduct(self):
        path = f"./log/{self.dataset_name}_whitening.json"
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
        return best_reduct
    
    send_success = pyqtSignal()
    send_best_reduct = pyqtSignal(int)
    send_ap = pyqtSignal(str, float)
    def train_start(self):
        args = default_argument_parser().parse_args()
        args.num_gpus = self.gpu

        for _ass in self.assurance_list:
            config_path = f"./training/configs/{self.dataset_name}/assurance_{_ass}.yaml"    
            args.config_file = config_path
            cfg = setup(args)
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

            cfg_wandb = yaml.safe_load(cfg.dump())
            wandb.init( project=f'ASS_{self.dataset_name}', name=f"assurance_{_ass}",config=cfg_wandb) 
            
            launch(training, num_gpus_per_machine=self.gpu, num_machines=1, machine_rank=0, dist_url=args.dist_url, args=(cfg, self.reduct, self.reduct_path, self.data_path),)
            wandb.finish()
            global trainer 

            evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], )
            val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
            results = trainer.test(cfg, model=trainer.model, evaluators=[evaluator])
            
            ap = results['box_proposals']['AR@1000']
            self.send_ap.emit(_ass, ap)
        self.send_best_reduct.emit(self.reduct)
        self.send_success.emit()
       

    

    
        
