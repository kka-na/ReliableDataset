from pathlib import Path
import json
import random
import imagesize
import os
import glob
import shutil
from PyQt5.QtCore import *

def create_image_annotation(file_path: Path, width: int, height: int, image_id: int):
    file_path = file_path.name
    image_annotation = {
        "file_name": file_path,
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation


def create_annotation_from_yolo_format(
    min_x, min_y, width, height, image_id, category_id, annotation_id, segmentation=True
):
    bbox = (float(min_x), float(min_y), float(width), float(height))
    area = width * height
    max_x = min_x + width
    max_y = min_y + height
    if segmentation:
        seg = [[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]]
    else:
        seg = []
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        "segmentation": seg,
    }

    return annotation


class DataSetting(QObject):
    def __init__(self, info):
        super(DataSetting, self).__init__()
        self.dataset_name = info[0]
        self.iter = info[1]
        self.init_path() 

    def init_path(self):
        self.base_path = f"/home/kana/Documents/Dataset/{self.dataset_name}"
        self.iter_path = f"{self.base_path}/deleting/iter{self.iter}"
        self.data_path = f"{self.base_path}/data/"
        self.sub_list = ['a','b','c','eval']
        self.type_list = ['train', 'val']
       
    def create_dir(self):
        extensions = ('*.png', '*.jpeg', '*.jpg')
        if os.path.exists(self.iter_path):
            shutil.rmtree(self.iter_path)  
        os.mkdir(self.iter_path)

        def check_and_make(path):
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                pass
        
        check_and_make(f"{self.base_path}/deleting/")
        check_and_make("./training/")
        check_and_make("./training/configs")
        check_and_make("./training/output")
        check_and_make(f"./training/configs/{self.dataset_name}")
        check_and_make(f"./training/output/{self.dataset_name}")
        check_and_make(f"{self.base_path}/whitening/")
        
        if self.iter == "1":
            image_paths = []
            for ext in extensions:
                for file_path in glob.glob(os.path.join(self.data_path, ext)):
                    image_paths.append(file_path)
            with open(f"{self.iter_path}/data.txt", 'w') as f:
                f.write('\n'.join(image_paths))
        else:
            data_cleaned_txt = f"{self.base_path}/deleting/iter{int(self.iter)-1}/data_deleted.txt"
            shutil.copy(data_cleaned_txt, f"{self.iter_path}/data.txt")

    
    send_data_num = pyqtSignal(str, int)
    send_success = pyqtSignal()
    def data_setting(self):
        
        with open(f"{self.base_path}/classes.txt", "r") as f:
            self.classes = [line.rstrip() for line in f.readlines()]

        with open(f"{self.iter_path}/data.txt", "r") as f:
            self.data_list = [line.strip() for line in f]

        #Division to Train & Validation
        random.seed(1)
        shuffled = random.sample(self.data_list, len(self.data_list))
        separation_size = len(self.data_list)//3

        split_data = {'a': shuffled[:separation_size],
                      'b': shuffled[separation_size:2*separation_size],
                      'c': shuffled[2*separation_size:],
                      'eval': shuffled}
        
        def write_txt(a, data_list):
            with open(f"{self.iter_path}/{a}.txt", 'w') as f:
                f.writelines(f"{file}\n" for file in data_list)

        def split(a, data_list):
            random.seed(1)
            shuffled = random.sample(data_list, len(data_list))
            split_size = int(len(shuffled) * 0.8)
            train_data, val_data = shuffled[:split_size], shuffled[split_size:]
            if a != 'eval':
                self.send_data_num.emit(a, split_size)
            else:
                self.send_data_num.emit(a, len(shuffled))
            
            write_txt(f"{a}_train", train_data)
            write_txt(f"{a}_val", val_data)

        for a, data_list in split_data.items():
            write_txt(a, data_list)
            split(a, data_list)

        #CONVERT to COCO Format
        for _sub in self.sub_list:
            for _type in self.type_list:
                coco_format = {"images": [{}], "categories": [], "annotations": [{}]}
                in_path = f"{self.iter_path}/{_sub}_{_type}.txt"
                (coco_format["images"],coco_format["annotations"],) = self.get_images_info_and_annotations(in_path)

                for index, label in enumerate(self.classes):
                    categories = {
                        "supercategory": "Defect",
                        "id": index + 1,  # ID starts with '1' .
                        "name": label,
                    }
                    coco_format["categories"].append(categories)
                out_path =f"{self.iter_path}/{_sub}_{_type}.json" 
                with open(out_path, "w") as outfile:
                    json.dump(coco_format, outfile, indent=4)
        self.send_success.emit()

    def get_images_info_and_annotations(self, path):
        path = Path(path)
        annotations = []
        images_annotations = []
        if path.is_dir():
            file_paths = sorted(path.rglob("*.jpg"))
            file_paths += sorted(path.rglob("*.jpeg"))
        else:
            with open(path, "r") as fp:
                read_lines = fp.readlines()
            file_paths = [Path(line.replace("\n", "")) for line in read_lines]

        image_id = 0
        annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

        for file_path in file_paths:
            w, h = imagesize.get(str(file_path))
            image_annotation = create_image_annotation(
                file_path=file_path, width=w, height=h, image_id=image_id
            )
            images_annotations.append(image_annotation)

            label_file_name = f"{file_path.stem}.txt"
            
            annotations_path = file_path.parent / label_file_name

            if not annotations_path.exists():
                continue  # The image may not have any applicable annotation txt file.

            with open(str(annotations_path), "r") as label_file:
                label_read_line = label_file.readlines()

            for line1 in label_read_line:
                label_line = line1
                category_id = (
                    int(label_line.split()[0]) + 1
                )  # you start with annotation id with '1'
                x_center = float(label_line.split()[1])
                y_center = float(label_line.split()[2])
                width = float(label_line.split()[3])
                height = float(label_line.split()[4])

                float_x_center = w * x_center
                float_y_center = h * y_center
                float_width = w * width
                float_height = h * height

                min_x = int(float_x_center - float_width / 2)
                min_y = int(float_y_center - float_height / 2)
                width = int(float_width)
                height = int(float_height)

                annotation = create_annotation_from_yolo_format(
                    min_x,
                    min_y,
                    width,
                    height,
                    image_id,
                    category_id,
                    annotation_id,
                )
                annotations.append(annotation)
                annotation_id += 1

            image_id += 1  # if you finished annotation work, updates the image id.

        return images_annotations, annotations
