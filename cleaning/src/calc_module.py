import os 
from operator import itemgetter

width = 1920
height = 1080

def get_confs(net_bboxes, checked_net):
    confs = []
    for i in range(len(net_bboxes)):
        if i in checked_net:
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


def calc_ious(gt_bboxes, net_bboxes):
    ious = []
    checked_gt = []
    checked_net = []
    for i in range(len(net_bboxes)):
        each_ious = {}
        for j in range(len(gt_bboxes)):
            iou = {int(j): float(calc_iou(gt_bboxes[j], net_bboxes[i]))}
            each_ious.update(iou)
        each_ious = sorted(each_ious.items(), key=lambda x: x[1], reverse=True)
        iou = 0.0
        for k in range(len(each_ious)):
            if each_ious[k][0] in checked_gt:
                continue
            else:
                iou = each_ious[k][1]
                checked_gt.append(each_ious[k][0])
                ious.append({int(net_bboxes[i]['cls']): float(iou)})
                checked_net.append(i)
                break
    return ious, checked_gt, checked_net

def calc_boxes(_object):
    lx = (float(_object[0]) - float(_object[2]) / 2) * width
    ly = (float(_object[1]) - float(_object[3]) / 2) * height
    rx = (float(_object[0]) + float(_object[2]) / 2) * width
    ry = (float(_object[1]) + float(_object[3]) / 2) * height
    return [float(_object[2]) * width, float(_object[3]) * height, lx, ly, rx, ry]

def get_label_list(_type, file):
    bboxes = []
    if os.path.isfile(file):
        fr = open(file)
        lines = fr.readlines()
        for line in lines:
            val = line.split()
            if _type == 0:
                conf = 1.0
                calc_box = calc_boxes(val[1:])
                _center = [float(val[1]) * width, float(val[2]) * height]
            else:
                conf = float(val[1])
                calc_box = calc_boxes(val[2:])
                _center = [float(val[2]) * width, float(val[3]) * height]
            bbox = {'cls': val[0], 'conf': conf, 'size': calc_box[0:2], 'bbox': calc_box[2:], 'center': _center}
            bboxes.append(bbox)

    bboxes = sorted(bboxes, key=itemgetter('conf'), reverse=True)
    return bboxes

def calc_each_score(gt_path, inf_path):
    gt_bboxes = get_label_list(0, gt_path)
    net_bboxes = get_label_list(1, inf_path)
    ious, checked_gt, checked_net = calc_ious(gt_bboxes, net_bboxes)
    confs = get_confs(net_bboxes, checked_net)
    classes = get_classes(gt_bboxes, checked_gt, ious)
    conf = 0
    correct_cnt = 0
    score = 0 
    for i in range(len(confs)):
        if classes[i]:
            score += confs[i]+float(list(ious[i].values())[0])
            correct_cnt += 1

    return score/(correct_cnt+1e-10)

