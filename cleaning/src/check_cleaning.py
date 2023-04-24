import cv2
import os
import sys
import calc_module
import random

class CheckCleaning():
    def __init__(self):
        super(CheckCleaning, self).__init__ ()
        self.iter=1
        self.daset_name = "TS"
        self.subset = "a"
        self.target1 = "b"
        self.target2 = "c"
        self.display = False
    
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

        self.base_path = "/home/kana/Documents/Dataset/{}".format(self.dataset_name)
        self.data_train_path = self.base_path+"/cleaning/iter{}/{}/".format(self.iter, self.subset)
        self.cleaning_txt = self.data_train_path+"cleaning_list.txt"
        self.cleaning_sample_path = self.base_path+"/cleaning_sample/iter{}/".format(self.iter)


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
        img_path = str(self.base_path+"/data/{}.jpg".format(file_name))
        if not os.path.exists(img_path):
            img_path = str(self.base_path+"/data/{}.png".format(file_name))
        image = cv2.imread(img_path, 1)
        h,w,c = image.shape
        gt_label = self.base_path+"/data/{}.txt".format(file_name)
        inf1_label = self.data_train_path+"inference_{}/{}.txt".format(self.target1, file_name)
        inf2_label = self.data_train_path+"inference_{}/{}.txt".format(self.target2, file_name)

        gt,net1 = calc_module.get_each_score_result_bbox(0.0, gt_label, inf1_label, w, h)
        gt, net2 = calc_module.get_each_score_result_bbox(0.0, gt_label, inf2_label, w, h)

        gt_image = self.get_result(image, gt)
        inf1_image = self.get_result(image,net1)
        inf1_score = calc_module.calc_score(gt_label, inf1_label, w, h)
        cv2.putText(inf1_image, str(inf1_score), (5,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        inf2_image = self.get_result(image, net2)
        inf2_score = calc_module.calc_score(gt_label, inf2_label, w, h)
        cv2.putText(inf2_image, str(inf2_score), (5,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

        image = cv2.hconcat([gt_image, inf1_image, inf2_image])
        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)),interpolation = cv2.INTER_AREA)
        
        #Display
        if self.display:
            cv2.imshow(file_name, image)
            cv2.waitKey(0)
        else:
            img_path = self.cleaning_sample_path+file_name+".jpg"
            if not os.path.exists(img_path):
                img_path = self.cleaning_sample_path+file_name+'.png'
            cv2.imwrite(img_path, image)
    
    def get_result(self, image,bboxes):
        image_cp = image.copy()
        for i, di in enumerate(bboxes):
            ll = list(di.items())[0]
            color = calc_module.get_color(int(ll[0]))
            name = str(i) + " " +calc_module.get_name(int(ll[0]))
            cv2.putText(image_cp, name, (int(ll[1][0]),int(ll[1][1])-2),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA )
            cv2.rectangle(image_cp, (int(ll[1][0]),int(ll[1][1])), (int(ll[1][2]),int(ll[1][3])),color, 3)
        return image_cp

if __name__ == "__main__":
    if len(sys.argv) !=4 :
        print("[Usage]: python3 check_cleaning.py datset_name iter# subset#")
        sys.exit()
    cc = CheckCleaning()
    cc.dataset_name = str(sys.argv[1])
    cc.iter = int(sys.argv[2])
    cc.subset = str(sys.argv[3])
    cc.check_cleaning()