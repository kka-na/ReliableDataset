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

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


# logging.getLogger("detectron2").setLevel(logging.WARNING)
# logging.getLogger("detectron2").propagate = False


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def training(cfg, ass, at, data_path, assurance_path):
    train_data = f"assurance_{at}_train"
    val_data = f"assurance_{at}_val"
    register_coco_instances(train_data, {}, f"{assurance_path}/{at}_train.json", data_path)
    register_coco_instances(val_data, {}, f"{assurance_path}/{at}_val.json", data_path)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], )
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    results = trainer.test(cfg, model=trainer.model, evaluators=[evaluator])
    
    if len(results) > 0:
        ap = results['bbox']['AP50']
        with open(f"{assurance_path}/ap_{at}_{ass}.txt", 'w') as f:
            f.write(str(ap))
    
class TrainStart(QObject):
    def __init__(self, info):
        super(TrainStart, self).__init__()
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
        self.gpu = 3

    send_success = pyqtSignal()
    
    send_ap = pyqtSignal(str, str, float)
    def train_start(self):
        args = default_argument_parser().parse_args()
        args.num_gpus = self.gpu

        for _at in ["before", "after"]:
            for _ass in self.assurance_list:
                config_path = f"./training/configs/{self.dataset_name}/assurance_{_at}_{_ass}.yaml"    
                args.config_file = config_path
                cfg = setup(args)
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

                cfg_wandb = yaml.safe_load(cfg.dump())
                wandb.init( project=f'ASS_{self.dataset_name}', name=f"assurance_{_at}_{_ass}",config=cfg_wandb) 
                
                launch(training, num_gpus_per_machine=self.gpu, num_machines=1, machine_rank=0, dist_url=args.dist_url, args=(cfg,  _ass, _at, self.data_path, self.assurance_path),)
                wandb.finish()

                with open(f"{self.assurance_path}/ap_{_at}_{_ass}.txt", 'r') as f:
                    line = f.readlines()[0]
                    ap = float(line.strip())
                    print(ap)
                self.send_ap.emit(_at, _ass, ap)

        self.send_success.emit()
       

    

    
        
