import os
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

from PyQt5.QtCore import pyqtSignal, QObject
import wandb
import yaml
import logging

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def training(cfg, reduction, reduct_path, reduct_90_path, data_path):
    
    train_data = f"reduct{reduction}_train"
    val_data = "reduct90_val"
    register_coco_instances(train_data, {}, f"{reduct_path}/train.json", data_path)
    register_coco_instances(val_data, {}, f"{reduct_90_path}/val.json", data_path)
            
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], )
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    results = trainer.test(cfg, model=trainer.model, evaluators=[evaluator])
    if len(results) > 0:
        ap = results['box_proposals']['AR@1000']
        with open(f"{reduct_path}/ap.txt", 'w') as f:
            f.write(str(ap))
    
class TrainStart(QObject):
    def __init__(self, info):
        super(TrainStart, self).__init__()
        self.dataset_name = info[0]
        self.reduct = info[2]
        self.init_path() 

        logging.getLogger("detectron2").setLevel(logging.WARNING)
        logging.getLogger("detectron2").propagate = False


    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        with open(f"{self.base_path}/classes.txt", "r") as f:
            classes = [line.rstrip() for line in f.readlines()]
        self.class_num = len(classes)
        whitening_path = f"{self.base_path}/whitening"
        self.reduct_path = f"{whitening_path}/reduct{self.reduct}"
        self.reduct_90_path = f"{whitening_path}/reduct90"
        self.data_path = f"{self.base_path}/data/"
        self.gpu = 3

    send_success = pyqtSignal()
    send_ap = pyqtSignal(float)
    def train_start(self):
        args = default_argument_parser().parse_args()
        args.num_gpus = self.gpu

        config_path = f"./training/configs/{self.dataset_name}/reduct{self.reduct}.yaml"    
        args.config_file = config_path
        cfg = setup(args)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        cfg_wandb = yaml.safe_load(cfg.dump())
        wandb.init( project=f'WHT_{self.dataset_name}', name=f"reduct{self.reduct}",config=cfg_wandb) 
        
        launch(training, num_gpus_per_machine=self.gpu, num_machines=1, machine_rank=0, dist_url=args.dist_url, args=(cfg, self.reduct, self.reduct_path, self.reduct_90_path, self.data_path),)
        wandb.finish()

        self.send_success.emit()
        with open(f"{self.reduct_path}/ap.txt", 'r') as f:
            line = f.readlines()[0]
            ap = float(line.strip())
            print(ap)
        self.send_ap.emit(ap)