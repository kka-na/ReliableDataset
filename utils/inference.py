import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm 
import multiprocessing
import concurrent.futures
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import cv2
import torch

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'

class Inference():
    def __init__(self, info):
        super(Inference, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/cleaning/iter{self.iter}"
        self.data_path = f"{self.base_path}/data"
        self.sub_list = ['a','b','c']
        self.gpu = 3
        self.cuda_devices = {'a':0, 'b':1, 'c':2}

    
    def init_cfg(self, _sub, _model):               
        cfg = get_cfg()
        cfg.merge_from_file(f"./training/output/{self.dataset_name}/iter{self.iter}/{_sub}/config.yaml")
        cfg.DATALOADER.NUM_WORKERS = 12
        cfg.MODEL.WEIGHTS = f"./training/output/{self.dataset_name}/iter{self.iter}/{_model}/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = torch.device(f"cuda:{self.cuda_devices[_sub]}")
        return cfg

    def inf_process(self, _sub, _model, target_path, inferenced_path):
        cfg = self.init_cfg(_sub, _model)
        predictor = DefaultPredictor(cfg)
        #predictor.model = torch.nn.DataParallel(predictor.model)
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

    def multi_process_val(self, _sub):
        val_path = f"{self.iter_path}/{_sub}_val.txt"
        inferenced_val_path = f"{self.iter_path}/{_sub}/val_inference/"
        os.makedirs(inferenced_val_path, exist_ok=True)
        self.inf_process(_sub, _sub, val_path, inferenced_val_path)

    def multi_process_sub_a(self, _model):
        train_path = f"{self.iter_path}/a_train.txt"
        with tf.device('/GPU:0'):
            inferenced_val_path = f"{self.iter_path}/a/inference_{_model}/"
            os.makedirs(inferenced_val_path, exist_ok=True)
            self.inf_process('a', _model, train_path, inferenced_val_path)
    
    def multi_process_sub_b(self, _model):
        train_path = f"{self.iter_path}/b_train.txt"
        with tf.device('/GPU:1'):
            inferenced_val_path = f"{self.iter_path}/b/inference_{_model}/"
            os.makedirs(inferenced_val_path, exist_ok=True)
            self.inf_process('b', _model, train_path, inferenced_val_path)

    def multi_process_sub_c(self, _model):
        train_path = f"{self.iter_path}/c_train.txt"
        with tf.device('/GPU:2'):   
            inferenced_val_path = f"{self.iter_path}/c/inference_{_model}/"
            os.makedirs(inferenced_val_path, exist_ok=True)
            self.inf_process('c', _model, train_path, inferenced_val_path)

    def inference(self):
        if self.gpu > 1:
            processes = []
            for _sub in self.sub_list:
                p = multiprocessing.Process(target=self.multi_process_val, args=(_sub,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(self.multi_process_sub_a, ['b', 'c'])
                executor.map(self.multi_process_sub_b, ['a', 'c'])
                executor.map(self.multi_process_sub_c, ['a', 'b'])


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
                        self.inf_process(_sub, __sub, train_path, inferenced_val_path)