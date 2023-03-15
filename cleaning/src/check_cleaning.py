import cv2
import os
import sys
import calc_module
import random

class CheckCleaning():
    def __init__(self):
        super(CheckCleaning, self).__init__ ()
        self.iter=0
        self.subset = "a"
        self.target1 = "b"
        self.target2 = "c"

        self.width = 1920
        self.height = 1080
        self.display = True
    
    def set_values(self):
        if self.subset == "a":
            self.target1 = "b"
            self.target2 = "c"
        elif self.subset == "b":
            self.target1 = "a"
            self.target2 = "c"
        elif self.subset == "c":
            self.target1 = "a"
            self.target2 = "b"

        base_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/cleaning/iter{}/".format(self.iter)
        self.data_train_path = base_path+"{}/train/".format(self.subset)
        self.cleaning_txt = self.data_train_path+"cleaning_list.txt"
        self.cleaning_sample_path = base_path+"cleaning_sample/"

        os.makedirs(self.cleaning_sample_path, exist_ok=True)

    def check_cleaning(self):
        self.set_values()
        f = open(self.cleaning_txt, 'r')
        lines = f.readlines()
        randomlist = random.sample(range(1, len(lines)-1), 10)
        for i in randomlist:
            self.disp_result(lines[i].strip())
            cv2.destroyAllWindows()
    
    def disp_result(self, file_name):
        image = cv2.imread(str(self.data_train_path+"data/{}.jpg".format(file_name)), 1)
        gt_label = self.data_train_path+"data/{}.txt".format(file_name)
        inf1_label = self.data_train_path+"inference_{}/{}.txt".format(self.target1, file_name)
        inf2_label = self.data_train_path+"inference_{}/{}.txt".format(self.target2, file_name)

        gt_image = self.get_result(image, 0, gt_label)
        inf1_image = self.get_result(image, 1, inf1_label)
        inf1_score = calc_module.calc_score(False, gt_label, inf1_label)
        cv2.putText(inf1_image, str(inf1_score), (5,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        inf2_image = self.get_result(image, 1, inf2_label)
        inf2_score = calc_module.calc_score(False, gt_label, inf2_label)
        cv2.putText(inf2_image, str(inf2_score), (5,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

        image = cv2.hconcat([gt_image, inf1_image, inf2_image])
        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)),interpolation = cv2.INTER_AREA)
        
        #Display
        if self.display:
            cv2.imshow(file_name, image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(self.cleaning_sample_path+file_name+".jpg", image)
    
    def get_result(self, image, _type, label):
        image_cp = image.copy()
        bboxes = calc_module.get_label_list(_type, label)
        cnt = 0
        for bbox in bboxes:
            color = self.get_color(bbox['cls'])
            name = str(cnt) + " " +self.get_name(bbox['cls'])
            cv2.putText(image_cp, name, (int(bbox['bbox'][0]),int(bbox['bbox'][1])-2),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA )
            cv2.rectangle(image_cp, (int(bbox['bbox'][0]),int(bbox['bbox'][1])), (int(bbox['bbox'][2]),int(bbox['bbox'][3])),color, 3)
            cnt += 1
        return image_cp

    def get_color(self, _cls): #BGR, Pastel Rainbow Colors
        colors = [[179,119,153],[184,137,216],[171,152,237],[142,184,243],[142,214,247],
                    [161,249,250],[150,221,195],[192,211,154],[225,209,140],[223,183,141]]
        return colors[int(_cls)]
    
    def get_name(self, _cls):
        names = ['person', 'bicycle', 'car','motorcycle','special vehicle','bus','-','truck', 'traffic sign','traffic light']
        return names[int(_cls)]


if __name__ == "__main__":
    if len(sys.argv) !=3 :
        print("[Usage]: python3 check_cleaning.py iter# subset#")
        sys.exit()
    cc = CheckCleaning()
    cc.iter = int(sys.argv[1])
    cc.subset = str(sys.argv[2])
    cc.check_cleaning()