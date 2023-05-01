import os
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from PyQt5.QtCore import *
import wandb
import yaml
import logging


class TrainStart(QObject):
    def __init__(self, info):
        super(TrainStart, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data/"
        self.sub_list = ['a','b','c','eval']

    send_success = pyqtSignal()
    def train_start(self):
        args = default_argument_parser().parse_args()
        args.num_gpus = 1

        for _sub in self.sub_list:
            self.subset = _sub
            config_path = f"./training/configs/{self.dataset_name}/iter{self.iter}/{_sub}.yaml"    
            args.config_file = config_path
            launch(self.training,args.num_gpus,args=(args,),)
        self.send_success.emit()
    
    def setup(self, args):
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)

        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)

        return cfg
    
    send_ap = pyqtSignal(str, float)
    def training(self, args):
        cfg = self.setup(args)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        #cfg_wandb = yaml.safe_load(cfg.dump())
       # wandb.init( project=f'NLC_{self.dataset_name}_{self.subset}', name=f"iter{self.iter}",config=cfg_wandb) 
        logging.getLogger("detectron2").setLevel(logging.WARNING)
        logging.getLogger("detectron2").propagate = False

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()
        evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], )
        val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        results = trainer.test(cfg, model=trainer.model, evaluators=[evaluator])
        if self.subset == 'eval':
            self.send_ap.emit(self.subset, results['box_proposals']['AR@1000'])
        else:
            self.send_ap.emit(self.subset, results['bbox']['AP50'])

        #wandb.finish()
        
