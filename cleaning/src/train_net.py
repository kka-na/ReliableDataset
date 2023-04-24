import logging
import os
from collections import OrderedDict

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
import yaml
import wandb

import sys



def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))     
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    output_dir = str(cfg.OUTPUT_DIR).split('/')
    dataset_name = output_dir[2]
    iter_num = output_dir[3][4:]
    subset=output_dir[4]

    base_path = "/home/kana/Documents/Dataset/{}/".format(dataset_name)
    data_path = base_path+"data/"
    iter_path = base_path+"cleaning/iter{}/".format(iter_num)
    train_data = "iter{}_{}_train".format(iter_num, subset)
    val_data = "iter{}_{}_val".format(iter_num, subset)

    register_coco_instances(train_data, {}, iter_path+"{}_train.json".format(subset), data_path)
    register_coco_instances(val_data, {}, iter_path+"{}_val.json".format(subset), data_path)


    cfg_wandb = yaml.safe_load(cfg.dump())
    wandb.init(project='NLC_{}_{}'.format(dataset_name, subset), name="iter{}".format(iter_num), config=cfg_wandb, sync_tensorboard=True) 

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    print("Train Over Successfully.")
    sys.exit(0)