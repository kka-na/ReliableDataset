import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm 
import concurrent.futures
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import cv2


class Inference():
    def __init__(self, info):
        super(Inference, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data/"
        self.sub_list = ['a','b','c']
        self.gpu = 3
        self.cuda_devices = [0,1,2]

    
    def init_cfg(self, _sub, _model):               
        cfg = get_cfg()
        cfg.merge_from_file(f"./training/output/{self.dataset_name}/iter{self.iter}/{_sub}/config.yaml")
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = f"./training/output/{self.dataset_name}/iter{self.iter}/{_sub}/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        return cfg

    def inf_process(self, _sub, _model, target_path, inferenced_path):
        cfg = self.init_cfg(_sub, _model)
        predictor = DefaultPredictor(cfg)
        f = open(target_path, 'r')
        img_list = f.readlines()
        img_list = list(map(lambda s:s.strip(), img_list))
        for img in tqdm(img_list):
            im = cv2.imread(img, 1)
            outputs = predictor(im)
            h,w,c = im.shape
            txt_path = inferenced_path + os.path.splitext(os.path.basename(img))[0]+'.txt'
            with open(txt_path, 'w') as f :
                num = int(outputs["instances"].pred_classes.size(0))
                for i in range(num):
                    pred_cls = str(int(outputs["instances"].pred_classes[i]))+' '
                    pred_score = str(float(outputs["instances"].scores[i]))+' '
                    pred_cx, pred_cy, pred_w, pred_h = '','','',''
                    for bbox in outputs["instances"].pred_boxes[i]:
                        box_w = float(bbox[2]-bbox[0])
                        box_h = float(bbox[3]-bbox[1])
                        pred_cx = str(float(bbox[0]+(box_w/2))/w)+' '
                        pred_cy = str(float(bbox[1]+(box_h/2))/h)+' '
                        pred_w = str(box_w/w)+' '
                        pred_h = str(box_h/h)+'\n'
                    pred_list = [pred_cls, pred_score, pred_cx, pred_cy, pred_w, pred_h]
                    f.writelines(pred_list)

    def process_val(self, _sub):
        with tf.device(f'/GPU:{self.cuda_devices[_sub % len(self.cuda_devices)]}'):
            val_path = f"{self.iter_path}/{_sub}_val.txt"
            inferenced_val_path = f"{self.iter_path}/{_sub}/val_inference/"
            os.makedirs(inferenced_val_path, exist_ok=True)
            self.inf_process(_sub, _sub, val_path, inferenced_val_path)

    def process_sub(self, _sub):
        with tf.device(f'/GPU:{self.cuda_devices[_sub % len(self.cuda_devices)]}'):
            train_path = f"{self.iter_path}/{_sub}_train.txt"
            for __sub in self.sub_list:
                if __sub != _sub:
                    inferenced_val_path = f"{self.iter_path}/{_sub}/inference_{__sub}/"
                    os.makedirs(inferenced_val_path, exist_ok=True)
                    self.inf_process(_sub, _sub, train_path, inferenced_val_path)

    def inference(self):
        if self.gpu != 1:
            #Val
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(self.process_val, self.sub_list)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(self.process_sub, self.sub_list)

        else:       
            #Inference Val
            for _sub in self.sub_list:
                val_path = f"{self.iter_path}/{_sub}_val.txt"
                inferenced_val_path = f"{self.iter_path}/{_sub}/val_inference/"
                os.makedirs(inferenced_val_path, exist_ok=True)
                self.inf_process(_sub, _sub, val_path, inferenced_val_path)
                    
            #Inference Subset
            for _sub in self.sub_list:
                train_path = f"{self.iter_path}/{_sub}_train.txt"
                for __sub in self.sub_list:
                    if __sub != _sub:
                        inferenced_val_path = f"{self.iter_path}/{_sub}/inference_{__sub}/"
                        os.makedirs(inferenced_val_path, exist_ok=True)
                        self.inf_process(_sub, _sub, train_path, inferenced_val_path)