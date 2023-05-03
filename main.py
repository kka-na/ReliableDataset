#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="A NumPy version")

import json
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.mainwindow import Ui_MainWindow
from utils import data_setting, deleting_train_setting, deleting_train_start, score_ensemble, cleaning, check_cleaning, calc_densities


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.button_connect()
        self.init_infos()

    def button_connect(self):
        self.ui.data_comboBox.currentTextChanged.connect(self.init_infos)
        self.ui.iter_comboBox.currentTextChanged.connect(self.init_infos)
        self.ui.initializeButton.clicked.connect(self.initialize)
        self.ui.startButton.clicked.connect(self.start)
        self.ui.pushButton.clicked.connect(self.do_data_setting)
        self.ui.pushButton_5.clicked.connect(self.do_train)
        self.ui.pushButton_6.clicked.connect(self.do_score_ensemble)
        self.ui.pushButton_7.clicked.connect(self.do_deleting)
        self.ui.pushButton_8.clicked.connect(self.do_check_deleting)
        self.ui.pushButton_4.clicked.connect(self.do_calc_densities)
        self.ui.pushButton_2.clicked.connect(self.go_sample_prev)
        self.ui.pushButton_3.clicked.connect(self.go_sample_next)
    
    def init_infos(self):
        self.dataset_name = str(self.ui.data_comboBox.currentText())
        self.iter = str(self.ui.iter_comboBox.currentText())
        self.info = (self.dataset_name, self.iter)


    def initialize(self):
        self.ds = data_setting.DataSetting(self.info)
        self.class_num = 0
        self.tsi = deleting_train_setting.TrainSetting(self.info)
        self.ts = deleting_train_start.TrainStart(self.info)
        self.se = score_ensemble.ScoreEnsemble(self.info)
        self.c = cleaning.Cleaning(self.info)
        self.cc = check_cleaning.CheckCleaning(self.info)
        self.cd = calc_densities.CalcDensities(self.info)
        self.random_samples = []
        self.sample_index = 0

        self.log_iter = {
                "a":{"NumOfData":0,"NumOfDelete":0,"Accuracy":0, "ScoreTh":0},
                "b":{"NumOfData":0,"NumOfDelete":0,"Accuracy":0, "ScoreTh":0},
                "c":{"NumOfData":0,"NumOfDelete":0,"Accuracy":0, "ScoreTh":0},
                "eval":{"NumOfData":0, "NumOfDelete":0, "Accuracy":0, "DeletingRatio":0} #NumOfWhole
            }

        self.set_connect()
        self.set_log_info()
        self.ui.initializeButton.setStyleSheet("")
        self.set_enabled()


    def set_connect(self):
        self.ds.send_data_num.connect(self.set_data_num)
        self.ds.send_success.connect(self.ds_success)
        self.ts.send_ap.connect(self.set_ap)
        self.ts.send_success.connect(self.ts_success)
        self.se.send_score.connect(self.set_score)
        self.ts.send_success.connect(self.ts_success)
        self.se.send_score.connect(self.set_score)
        self.se.send_deleted.connect(self.set_deleted)
        self.se.send_success.connect(self.se_success)
        self.c.send_ratio.connect(self.set_ratio)
        self.c.send_success.connect(self.c_success)
        self.cc.send_random_samples.connect(self.set_random_samples)
        self.cc.send_success.connect(self.cc_success)
        self.cd.send_success.connect(self.cd_success)

    def set_enabled(self):
        self.ui.startButton.setEnabled(True)
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_5.setEnabled(True)
        self.ui.pushButton_6.setEnabled(True)
        self.ui.pushButton_7.setEnabled(True)
        self.ui.pushButton_8.setEnabled(True)
        self.ui.pushButton_4.setEnabled(True)

    def set_log_info(self):
        path = f"./log/{self.dataset_name}.json"
        if not os.path.isfile(path):
            return
        else:
            with open(path, 'r') as f:
                data = json.load(f)
                if f"Iteration{self.iter}" in data:
                    log_iter = data[f"Iteration{self.iter}"] 
                    for sub in ['a', 'b', 'c', 'eval']:
                        self.set_data_num(sub, log_iter[f"{sub}"]["NumOfData"])
                        self.set_deleted(sub, log_iter[f"{sub}"]["NumOfDelete"])
                        self.set_ap(sub, log_iter[f"{sub}"]["Accuracy"])
                        if sub != 'eval':
                            self.set_score(sub, log_iter[f"{sub}"]["ScoreTh"])
                        else:
                            self.set_ratio(log_iter[f"{sub}"]["DeletingRatio"])
                else:
                    return
                

    def start(self):
        self.initialize()
        if self.ui.radioButton.isChecked():
            self.ds.create_dir()
        self.ds.data_setting()
        self.tsi.train_setting()
        self.ts.train_start()
        self.se.score_ensemble()
        self.c.cleaning()
        self.save_log()
        self.cc.check_cleaning()
        self.cd.calc_densities()
        
    def save_log(self):
        path = f"./log/{self.dataset_name}.json"
        if os.path.isfile(path):
            with open(path, 'r') as f:
                data = json.load(f)
            data[f'Iteration{self.iter}'] = self.log_iter
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            data = {"Dataset":self.dataset_name,f"Iteration{self.iter}":self.log_iter}
            log_json = json.dumps(data, ensure_ascii=False, indent=4)
            with open(path, 'w') as f:
                f.write(log_json)

    def do_data_setting(self):
        if self.ui.radioButton.isChecked():
            self.ds.create_dir()
        self.ds.data_setting()
    
    def do_train(self):
        self.tsi.train_setting()
        self.ts.train_start()
    
    def do_score_ensemble(self):
        self.se.score_ensemble()
    
    def do_deleting(self):
        self.c.cleaning()
    
    def do_check_deleting(self):
        self.cc.check_cleaning()

    def do_calc_densities(self):
        self.cd.calc_densities()

    @pyqtSlot(str,int)
    def set_data_num(self, sub, num):
        self.log_iter[f"{sub}"]["NumOfData"]=num
        if sub == 'a':
            self.ui.label_11.setText(str(num))
        elif sub == 'b':
            self.ui.label_21.setText(str(num))
        elif sub == 'c':
            self.ui.label_16.setText(str(num))
        elif sub == 'eval':
            self.ui.label_28.setText(str(num))

    @pyqtSlot(str, float)
    def set_ap(self, sub, ap):
        def AP(_ap):
            return str(round(ap*100,2))
        self.log_iter[f"{sub}"]["Accuracy"]=float(AP(ap))
        if sub == 'a':
            self.ui.label_13.setText(AP(ap))
        elif sub == 'b':
            self.ui.label_23.setText(AP(ap))
        elif sub == 'c':
            self.ui.label_18.setText(AP(ap))
        elif sub == 'eval':
            self.ui.label_36.setText(str(round(ap,2)))
    
    @pyqtSlot(str, float)
    def set_score(self, sub, sc):
        def SC(_sc):
            return str(round(sc*100,2))
        self.log_iter[f"{sub}"]["ScoreTh"]=float(SC(sc)) 
        if sub == 'a':
            self.ui.label_14.setText(SC(sc))
        elif sub == 'b':
            self.ui.label_24.setText(SC(sc))
        elif sub == 'c':
            self.ui.label_19.setText(SC(sc))
    
    @pyqtSlot(str, int)
    def set_deleted(self, sub, deleted):
        self.log_iter[f"{sub}"]["NumOfDelete"]=deleted
        if sub == 'a':
            self.ui.label_12.setText(str(deleted))
        elif sub == 'b':
            self.ui.label_22.setText(str(deleted))
        elif sub == 'c':
            self.ui.label_17.setText(str(deleted))
        elif sub == 'eval':
            self.ui.label_29.setText(str(deleted))
    
    @pyqtSlot(float)
    def set_ratio(self, ratio):
        self.ui.label_30.setText(str(round(ratio, 2))+"%")
        self.log_iter["eval"]["DeletingRatio"]=round(ratio, 2)

    def return_pixmap(self, index):
        sample_img = self.random_samples[index]
        h,w,ch = sample_img.shape
        bpl = ch * w
        q_image = QImage(sample_img.data, w,h,bpl, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    @pyqtSlot(object)
    def set_random_samples(self, samples):
        self.random_samples = samples
        self.ui.label_3.setPixmap(self.return_pixmap(self.sample_index))
    
    def go_sample_prev(self):
        self.sample_index = max(self.sample_index - 1, 0)
        self.ui.label_3.setPixmap(self.return_pixmap(self.sample_index))

    def go_sample_next(self):
        self.sample_index = min(self.sample_index + 1, len(self.random_samples)-1)
        self.ui.label_3.setPixmap(self.return_pixmap(self.sample_index))


    @pyqtSlot()
    def ds_success(self):
        self.ui.pushButton.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def ts_success(self):
        self.ui.pushButton_5.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def se_success(self):
        self.ui.pushButton_6.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def c_success(self):
        self.ui.pushButton_7.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def cc_success(self):
        self.ui.pushButton_8.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def cd_success(self):
        self.ui.pushButton_4.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())