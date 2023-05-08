import os 
import cv2
from math import sqrt
from operator import itemgetter

def get_confs(net_bboxes, checked_net):
    confs = []
    for i in checked_net:
        confs.append(net_bboxes[i]['conf'])
    return confs

def get_classes(gt_bboxes, checked_gt, ious):
    classes = []
    for i in range(len(ious)):
        if int(gt_bboxes[checked_gt[i]]['cls']) == int(list(ious[i].keys())[0]):
            classes.append(1)
        else:
            classes.append(0)
    return classes

def calc_mconf(bboxes):
    conf = 0.0
    for bbox in bboxes:
        conf += bbox['conf']
    return conf / (len(bboxes) + 1e-9)

def calc_mious(ious):
    iou_sum = 0.0
    for i in range(len(ious)):
        iou_sum += float(list(ious[i].values())[0])
    if iou_sum <= 0.0:
        miou = 0.0
    else:
        miou = iou_sum / float(len(ious) + 1e-9)
    return miou


def calc_iou(gt_bbox, net_bbox):
    x1 = max(gt_bbox['bbox'][0], net_bbox['bbox'][0])
    y1 = max(gt_bbox['bbox'][1], net_bbox['bbox'][1])
    x2 = min(gt_bbox['bbox'][2], net_bbox['bbox'][2])
    y2 = min(gt_bbox['bbox'][3], net_bbox['bbox'][3])
    tmp_width = x2 - x1
    tmp_height = y2 - y1
    iou = 0.0
    if tmp_width < 0 or tmp_height < 0:
        iou = 0.0
        return iou
    overlap = tmp_width * tmp_height
    a = (gt_bbox['bbox'][2] - gt_bbox['bbox'][0]) * (gt_bbox['bbox'][3] - gt_bbox['bbox'][1])
    b = (net_bbox['bbox'][2] - net_bbox['bbox'][0]) * (net_bbox['bbox'][3] - net_bbox['bbox'][1])
    combine = a + b - overlap
    iou = float(overlap / (combine + 1e-5))
    return iou


def calc_ious(iou_thres, gt_bboxes, net_bboxes):
    ious = []
    checked_gt = []
    checked_net = []
    for i in range(len(net_bboxes)):
        each_ious = {}
        for j in range(len(gt_bboxes)):
            iou = {int(j): float(calc_iou(gt_bboxes[j], net_bboxes[i]))}
            each_ious.update(iou)
        #Sorting by IoU
        each_ious = sorted(each_ious.items(), key=lambda x: x[1], reverse=True)
        iou = 0.0
        for k in range(len(each_ious)):
            iou = each_ious[k][1]
            if iou > iou_thres:
                checked_net.append(i)
                checked_gt.append(each_ious[k][0])
                ious.append({int(net_bboxes[i]['cls']): float(iou)})
    return ious, checked_gt, checked_net

def calc_boxes(_object, width, height):
    lx = (float(_object[0]) - float(_object[2]) / 2) * width
    ly = (float(_object[1]) - float(_object[3]) / 2) * height
    rx = (float(_object[0]) + float(_object[2]) / 2) * width
    ry = (float(_object[1]) + float(_object[3]) / 2) * height
    return [float(_object[2]) * width, float(_object[3]) * height, lx, ly, rx, ry]

def get_bbox_class_list(file):
    with open(file, 'r') as f:
        class_list = [line.split()[0] for line in f.readlines()]
    return class_list

def get_bbox_size_list(file, width, height):
    size_list = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            size_list.append(get_bbox_size_category(line.split(), width, height))
    return size_list

def get_bbox_cnt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        return len(lines)

def get_bbox_class_cnt_list(file, all_bbox_class):
    with open(file, 'r') as f :
        lines = f.readlines()
        for line in lines:
            value = line.split()
            all_bbox_class[int(value[0])] += 1

def get_bbox_size_category(bbox, width, height):
    weight = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    calc_box = calc_boxes(bbox[1:], width, height)
    volume = width*height
    size = calc_box[0]*calc_box[1]
    size_ind = 1
    for i in range(1,6):
        if (volume*weight[i-1]<size) and (size <= volume*weight[i]):
            size_ind = i
    return size_ind - 1
    
def get_bbox_size_cnt_list(file, all_bbox_size, width, height):
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            size_category = get_bbox_size_category(line.split(), width, height)
            all_bbox_size[size_category] += 1


def get_label_list(_type, file, width, height, conf_th = 0.7):
    bboxes = []
    if os.path.isfile(file):
        fr = open(file)
        lines = fr.readlines()
        for line in lines:
            val = line.split()
            if _type == 0:
                conf = 1.0
                calc_box = calc_boxes(val[1:], width, height)
                _center = [float(val[1]) * width, float(val[2]) * height]
            else:
                conf = float(val[1])
                calc_box = calc_boxes(val[2:], width, height)
                _center = [float(val[2]) * width, float(val[3]) * height]
            if conf >= conf_th:
                bbox = {'cls': val[0], 'conf': conf, 'size': calc_box[0:2], 'bbox': calc_box[2:], 'center': _center}
                bboxes.append(bbox)

    bboxes = sorted(bboxes, key=itemgetter('conf'), reverse=True)
    return bboxes

def calc_norm_variance(_list, cnt_factor, norm_factor):
    cnt_sum = 0.0
    for i in range(cnt_factor):
        cnt_sum += _list[i]
    avg = cnt_sum/float(cnt_factor) if cnt_sum != 0 else 0.0
    avg = avg/float(norm_factor) if avg != 0 else 0.0
    norm_list = [0.0]*cnt_factor
    for i in range(cnt_factor):
        norm_list[i] = float(_list[i])/float(norm_factor) if _list[i] != 0 else 0.0
    dev_sum = 0.0
    dev_list = [0.0]*cnt_factor
    for i in range(cnt_factor):
        dev = norm_list[i]-avg
        dev_sum += pow(dev,2)
        dev_list[i] = dev*-1 #Change the sign to give sparse objects a higher score
    var = float(dev_sum)/float(cnt_factor) if dev_sum != 0 else 0.0
    std_var = sqrt(var) if var!= 0 else 0.0
    return std_var, dev_list

def calc_z_score(cnt_factor, std_var, dev_list):
    z_scores = [0.0]*cnt_factor
    for i in range(cnt_factor):
        z_scores[i] = float(dev_list[i]/std_var)
    return z_scores

def get_color(_cls): #BGR, Pastel Rainbow Colors
    colors = [[179,119,153],[184,137,216],[171,152,237],[142,184,243],[142,214,247],
                [161,249,250],[150,221,195],[192,211,154],[225,209,140],[223,183,141]]
    return colors[int(_cls)]

def get_result(image, arr, classes):
    image_cp = image.copy()
    for i, di in enumerate(arr):
        ll = list(di.items())[0]
        color = get_color(int(ll[0]))
        name = f"{str(i)} {classes[int(ll[0])]}"
        cv2.putText(image_cp, name, (int(ll[1][0]),int(ll[1][1])-2),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
        cv2.rectangle(image_cp, (int(ll[1][0]),int(ll[1][1])), (int(ll[1][2]),int(ll[1][3])),color, 3)
    return image_cp

def get_test_img(im, test_bbox_gt, test_bbox_net):
    gt = get_result(im, test_bbox_gt)
    net = get_result(im, test_bbox_net)
    res = cv2.hconcat([gt, net])
    res = cv2.resize(res, (int(res.shape[1]/2), int(res.shape[0]/2)),interpolation = cv2.INTER_AREA)
    return res

def get_each_score_result_bbox(iou_thres, gt_path, inf_path, width, height):
    gt_bboxes = get_label_list(0,  gt_path,width, height)
    net_bboxes = get_label_list(1,inf_path, width, height)
    ious, checked_gt, checked_net = calc_ious(iou_thres, gt_bboxes, net_bboxes)
    confs = get_confs(net_bboxes, checked_net)
    classes = get_classes(gt_bboxes, checked_gt, ious)
    gt = []
    net = []
    conf = 0
    for i in range(len(confs)):
        conf = confs[i] if classes[i] else 0
        score = conf*float(list(ious[i].values())[0])
        if score > 0:
            gt.append({gt_bboxes[checked_gt[i]]['cls']:gt_bboxes[checked_gt[i]]['bbox']})
            net.append({net_bboxes[checked_net[i]]['cls']:net_bboxes[checked_net[i]]['bbox']})
    return gt, net

def get_gt_bbox(gt_path, width, height):
    gt_bboxes = get_label_list(0, gt_path, width, height)
    gt = []
    for i in range(len(gt_bboxes)):
        gt.append({gt_bboxes[i]['cls']:gt_bboxes[i]['bbox']})
    return gt


def calc_each_score(iou_thres, gt_path, inf_path, width, height, im):
    gt_bboxes = get_label_list(0,  gt_path,width, height)
    net_bboxes = get_label_list(1,inf_path, width, height)
    ious, checked_gt, checked_net = calc_ious(iou_thres, gt_bboxes, net_bboxes)
    confs = get_confs(net_bboxes, checked_net)
    classes = get_classes(gt_bboxes, checked_gt, ious)

    conf = 0
    correct_cnt = 0
    score_sum = 0 

    for i in range(len(confs)):
        conf = confs[i] if classes[i] else 0
        score = conf*float(list(ious[i].values())[0])
        score_sum += score
        correct_cnt += 1

    return score_sum/(correct_cnt+1e-10)

def calc_score_threshold(gt_path, inf_path, width, height, im=None):
    return calc_each_score(0.0, gt_path, inf_path, width, height, im)

def calc_score(gt_path, inf_path,width, height, im=None):
    return calc_each_score(0.3, gt_path, inf_path, width, height, im)