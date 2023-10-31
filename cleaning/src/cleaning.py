import sys
import os
from tqdm import tqdm

class Cleaning():
    def __init__(self):
        super(Cleaning, self).__init__()
        self.iter = 0
        self.dataset_name = "TS"
    
    def set_values(self):
        self.base_path = "/home/kana/Documents/Dataset/{}".format(self.dataset_name)
        self.data_path =  "{}/data".format(self.base_path)
        self.iter_path = "{}/deleting/iter{}".format(self.base_path, self.iter)
        self.cleaning_list = []
        self.cleaning_list.append("{}/{}/deleting_list.txt".format(self.iter_path, "a"))
        self.cleaning_list.append("{}/{}/deleting_list.txt".format(self.iter_path, "b"))
        self.cleaning_list.append("{}/{}/deleting_list.txt".format(self.iter_path, "c"))

    def cleaning(self):
        self.set_values()
        f_pre_list = []
        with open("{}/data.txt".format(self.iter_path), 'r') as f_pre:
            f_pre_list = set(line.strip() for line in f_pre)

        for file in self.cleaning_list:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    name = line.strip()
                    img = f"{self.data_path}/{name}.jpg"
                    if not os.path.exists(img):
                        img = f"{self.data_path}/{name}.png"
                    f_pre_list.discard(img)
        
        with open("{}/data_cleaned.txt".format(self.iter_path), 'w') as f_aft:
            for n in f_pre_list:
                f_aft.write("%s\n"%n)
            
        

if __name__=="__main__":
    if len(sys.argv) != 3 :
        print("[Usage]: python3 calc_conf_th.py dataset_name iter#")
        sys.exit()
    c = Cleaning()
    c.dataset_name = str(sys.argv[1])
    c.iter = int(sys.argv[2])
    c.cleaning()
    sys.exit(0)