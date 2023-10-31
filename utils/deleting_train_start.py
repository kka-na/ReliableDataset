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

global ap
def training(cfg, iter, iter_path, subset, data_path):
    train_data = f"iter{iter}_{subset}_train"
    val_data = f"iter{iter}_{subset}_val"
    register_coco_instances(train_data, {}, f"{iter_path}/{subset}_train.json", data_path)
    register_coco_instances(val_data, {}, f"{iter_path}/{subset}_val.json", data_path)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], )
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    results = trainer.test(cfg, model=trainer.model, evaluators=[evaluator])

    if len(results)>0:
        if subset == 'eval':
            ap = results['box_proposals']['AR@1000']
        else:
            ap = results['bbox']['AP50']
        with open(f"{iter_path}/ap_{subset}.txt", "w") as f:
            f.write(str(ap))

class TrainStart(QObject):
    def __init__(self, info):
        super(TrainStart, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/deleting/iter{self.iter}"
        self.data_path = f"{self.base_path}/data/"
        self.sub_list = ['a','b','c','eval']
        self.gpu = 3

    send_success = pyqtSignal()
    send_ap = pyqtSignal(str, float)
    def train_start(self):
        args = default_argument_parser().parse_args()
        args.num_gpus = self.gpu

        for _sub in self.sub_list:
            config_path = f"./training/configs/{self.dataset_name}/iter{self.iter}/{_sub}.yaml"    
            args.config_file = config_path
            cfg = setup(args)
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

            cfg_wandb = yaml.safe_load(cfg.dump())
            wandb.init( project=f'NLC_{self.dataset_name}_{_sub}', name=f"iter{self.iter}",config=cfg_wandb) 
            
            launch(training, num_gpus_per_machine=self.gpu, num_machines=1, machine_rank=0, dist_url=args.dist_url, args=(cfg, self.iter, self.iter_path, _sub, self.data_path),)
        
            wandb.finish()

            with open(f"{self.iter_path}/ap_{_sub}.txt", 'r') as f:
                line = f.readlines()[0]
                ap = float(line.strip())
                print(ap)

            self.send_ap.emit(_sub, ap)

        self.send_success.emit()

    

    
        
