import sys
from tqdm import tqdm

class Cleaning():
    def __init__(self):
        super(Cleaning, self).__init__()
        self.iter = 0
        self.base_path = "/home/kana/Documents/Dataset/TS/2DOD"
    
    def set_values(self):
        self.data_path =  "{}/data".format(self.base_path)
        self.iter_path = "{}/cleaning/iter{}".format(self.base_path, self.iter)
        self.cleaning_list = []
        self.cleaning_list.append("{}/{}/cleaning_list.txt".format(self.iter_path, "a"))
        self.cleaning_list.append("{}/{}/cleaning_list.txt".format(self.iter_path, "b"))
        self.cleaning_list.append("{}/{}/cleaning_list.txt".format(self.iter_path, "c"))

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
                    jpg = f"{self.data_path}/{name}.jpg"
                    f_pre_list.discard(jpg)
        
        with open("{}/data_cleaned.txt".format(self.iter_path), 'w') as f_aft:
            for n in f_pre_list:
                f_aft.write("%s\n"%n)
            
        

if __name__=="__main__":
    if len(sys.argv) != 2 :
        print("[Usage]: python3 calc_conf_th.py iter#")
        sys.exit()
    c = Cleaning()
    c.iter = int(sys.argv[1])
    c.cleaning()