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
from utils import deleting_data_setting, deleting_train_setting, deleting_train_start, score_ensemble, deleting, check_deleting, calc_densities, whitening_data_setting, whitening_train_setting, whitening_train_start, check_whitening, assurance_train_setting, assurance_train_start

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
        self.ui.comboBox.currentTextChanged.connect(self.init_infos)
        self.ui.initializeButton.clicked.connect(self.initialize)
        self.ui.startButton.clicked.connect(self.deleting_start)
        self.ui.pushButton_11.clicked.connect(self.whitening_start)
        self.ui.pushButton.clicked.connect(self.do_data_setting)
        self.ui.pushButton_5.clicked.connect(self.do_deleting_training)
        self.ui.pushButton_6.clicked.connect(self.do_score_ensemble)
        self.ui.pushButton_7.clicked.connect(self.do_deleting)
        self.ui.pushButton_8.clicked.connect(self.do_check_deleting)
        self.ui.pushButton_4.clicked.connect(self.do_calc_densities)
        self.ui.pushButton_10.clicked.connect(self.do_whitening_data_setting)
        self.ui.pushButton_9.clicked.connect(self.do_whitening_training)
        self.ui.pushButton_12.clicked.connect(self.do_check_whitening)
        self.ui.pushButton_14.clicked.connect(self.do_assurance_training)
        self.ui.pushButton_2.clicked.connect(self.go_sample_prev)
        self.ui.pushButton_3.clicked.connect(self.go_sample_next)
    
    def init_infos(self):
        self.dataset_name = str(self.ui.data_comboBox.currentText())
        self.iter = str(self.ui.iter_comboBox.currentText())
        self.reduct = int(self.ui.comboBox.currentText())
        self.info = (self.dataset_name, self.iter, self.reduct)

    def initialize(self):
        self.dds = deleting_data_setting.DataSetting(self.info)
        self.class_num = 0
        self.dtsi = deleting_train_setting.TrainSetting(self.info)
        self.dts = deleting_train_start.TrainStart(self.info)
        self.se = score_ensemble.ScoreEnsemble(self.info)
        self.d = deleting.Deleting(self.info)
        self.ckd = check_deleting.CheckDeleting(self.info)
        self.cd = calc_densities.CalcDensities(self.info)
        self.wds = whitening_data_setting.DataSetting(self.info)
        self.wtsi = whitening_train_setting.TrainSetting(self.info)
        self.wts = whitening_train_start.TrainStart(self.info)
        self.ckw = check_whitening.CheckWhitening(self.info)
        self.atsi = assurance_train_setting.TrainSetting(self.info)
        self.ats = assurance_train_start.TrainStart(self.info)

        self.random_samples = []
        self.sample_index = 0

        self.log_iter = {
            "a":{"NumOfData":0,"NumOfDelete":0,"Accuracy":0, "ScoreTh":0},
            "b":{"NumOfData":0,"NumOfDelete":0,"Accuracy":0, "ScoreTh":0},
            "c":{"NumOfData":0,"NumOfDelete":0,"Accuracy":0, "ScoreTh":0},
            "eval":{"NumOfData":0, "NumOfDelete":0, "Accuracy":0, "DeletingRatio":0} #NumOfWhole
        }
        self.log_reduct = {
            "NumOfData":0, "NumOfReduct":0, "Accuracy":0
        }

        self.set_connect()
        self.set_log_iter()
        self.ui.initializeButton.setStyleSheet("")
        self.set_enabled()


    def set_connect(self):
        self.dds.send_data_num.connect(self.set_data_num)
        self.dds.send_success.connect(self.dds_success)
        self.dts.send_ap.connect(self.set_ap)
        self.dts.send_success.connect(self.dts_success)
        self.se.send_score.connect(self.set_score)
        self.se.send_score.connect(self.set_score)
        self.se.send_deleted.connect(self.set_deleted)
        self.se.send_success.connect(self.se_success)
        self.d.send_ratio.connect(self.set_ratio)
        self.d.send_success.connect(self.d_success)
        self.ckd.send_random_samples.connect(self.set_random_samples)
        self.ckd.send_success.connect(self.ckd_success)
        self.cd.send_success.connect(self.cd_success)
        self.cd.send_data_num.connect(self.set_whitening_data_num)
        self.wds.send_success.connect(self.wds_success)
        self.wts.send_success.connect(self.wts_success)
        self.wts.send_ap.connect(self.set_whitening_ap)
        self.ckw.send_random_samples.connect(self.set_random_samples)
        self.ckw.send_success.connect(self.ckw_success)
        self.ats.send_success.connect(self.ats_success)
        self.ats.send_ap.connect(self.set_assurance_ap)
        self.ats.send_best_reduct.connect(self.set_best_reduct)

    def set_enabled(self):
        self.ui.startButton.setEnabled(True)
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_5.setEnabled(True)
        self.ui.pushButton_6.setEnabled(True)
        self.ui.pushButton_7.setEnabled(True)
        self.ui.pushButton_8.setEnabled(True)
        self.ui.pushButton_4.setEnabled(True)
        self.ui.pushButton_9.setEnabled(True)
        self.ui.pushButton_10.setEnabled(True)
        self.ui.pushButton_11.setEnabled(True)
        self.ui.pushButton_12.setEnabled(True)
        self.ui.pushButton_14.setEnabled(True)

    def set_log_iter(self):
        path = f"./log/{self.dataset_name}_deleting.json"
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
                

    def deleting_start(self):
        if self.ui.radioButton.isChecked():
            self.dds.create_dir()
        self.dds.data_setting()
        self.dtsi.train_setting()
        self.dts.train_start()
        self.se.score_ensemble()
        self.d.deleting()
        self.save_log_iter()
        self.ckd.check_deleting()
    
    def whitening_start(self):
        self.cd.calc_densities()
        self.wds.data_setting()
        self.wtsi.train_setting()
        self.wts.train_start()
        self.save_log_reduct()
        self.ckw.check_whitening()
    
    def assurance_start(self):
        self.atsi.train_setting()
        self.ats.train_start()

    def save_log_iter(self):
        path = f"./log/{self.dataset_name}_deleting.json"
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

    def save_log_reduct(self):
        path = f"./log/{self.dataset_name}_whitening.json"
        if os.path.isfile(path):
            with open(path, 'r') as f:
                data = json.load(f)
            data[f'Reduct{self.reduct}'] = self.log_reduct
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            data = {"Dataset":self.dataset_name, f"Reduct{self.reduct}":self.log_reduct}
            log_json = json.dumps(data, ensure_ascii=False, indent=4)
            with open(path, 'w') as f:
                f.write(log_json)

    def do_data_setting(self):
        if self.ui.radioButton.isChecked():
            self.dds.create_dir()
        self.dds.data_setting()
    
    def do_deleting_training(self):
        self.dtsi.train_setting()
        self.dts.train_start()
    
    def do_score_ensemble(self):
        self.se.score_ensemble()
    
    def do_deleting(self):
        self.d.deleting()
        self.save_log_iter()
    
    def do_check_deleting(self):
        self.ckd.check_deleting()

    def do_calc_densities(self):
        self.cd.calc_densities()
    
    def do_whitening_data_setting(self):
        self.wds.data_setting()

    def do_whitening_training(self):
        self.wtsi.train_setting()
        self.wts.train_start()
    
    def do_check_whitening(self):
        self.ckw.check_whitening()
    
    def do_assurance_training(self):
        self.atsi.train_setting()
        self.ats.train_start()

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
    
    @pyqtSlot(int, int)
    def set_whitening_data_num(self, before, after):
        self.log_reduct["NumOfData"]=before
        self.log_reduct["NumOfReduct"]=after
        self.ui.label_34.setText(str(before))
        self.ui.label_37.setText(str(after))

    @pyqtSlot(float)
    def set_whitening_ap(self, ap):
        self.log_reduct["Accuracy"]=round(ap,2)
        self.ui.label_38.setText(str(round(ap, 2)))
    
    @pyqtSlot(int)
    def set_best_reduct(self, reduct):
        self.ui.label_46.setText(str(reduct))

    @pyqtSlot(str, float)
    def set_assurance_ap(self, ass, ap):
        if ass == 'a':
            self.ui.label_43.setText(str(round(ap,2)))
        elif ass == 'b':
            self.ui.label_44.setText(str(round(ap,2)))
    
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
    def dds_success(self):
        self.ui.pushButton.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def dts_success(self):
        self.ui.pushButton_5.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def se_success(self):
        self.ui.pushButton_6.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def d_success(self):
        self.ui.pushButton_7.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def ckd_success(self):
        self.ui.pushButton_8.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def cd_success(self):
        self.ui.pushButton_4.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def wds_success(self):
        self.ui.pushButton_10.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def wts_success(self):
        self.ui.pushButton_9.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def ckw_success(self):
        self.ui.pushButton_12.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def ads_success(self):
        self.ui.pushButton_13.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()
    @pyqtSlot()
    def ats_success(self):
        self.ui.pushButton_14.setStyleSheet("QPushButton{color:rgb(255,255,255);background-color:rgb(0,0,0);}")
        QCoreApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())