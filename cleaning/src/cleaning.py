import glob
import sys
import os

class Cleaning():
    def __init__(self):
        super(Cleaning, self).__init__()
        self.iter = 0
        self.base_path = "/home/kana/Documents/Dataset/TS/2DOD/organized/"
    
    def set_values(self):
        self.data_path = self.base_path+"data/"
        self.label_path = self.base_path+"label/"
        self.cleaning_list = []
        self.cleaning_list.append(str(self.base_path+("cleaning/iter{}/{}/train/cleaning_list.txt".format(self.iter, "a"))))
        self.cleaning_list.append(str(self.base_path+("cleaning/iter{}/{}/train/cleaning_list.txt".format(self.iter, "b"))))
        self.cleaning_list.append(str(self.base_path+("cleaning/iter{}/{}/train/cleaning_list.txt".format(self.iter, "c"))))

    def cleaning(self):
        self.set_values()
        for file in self.cleaning_list:
            f = open(file, 'r')
            lines = f.readlines()
            for line in lines:
                name = line.split('\n')[0]
                img_path = self.data_path+name+".jpg"
                label_path = self.label_path+name+".txt"
                if os.path.isfile(img_path):
                    os.remove(img_path)
                if os.path.isfile(label_path):
                    os.remove(label_path)
            f.close()

if __name__=="__main__":
    if len(sys.argv) != 2 :
        print("[Usage]: python3 calc_conf_th.py iter#")
        sys.exit()
    c = Cleaning()
    c.iter = int(sys.argv[1])
    c.cleaning()