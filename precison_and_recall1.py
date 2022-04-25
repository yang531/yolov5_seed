import os
import json
import  numpy as np
from PIL import Image

# class name
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
# 初始化二维0数组
result_list = np.array(np.zeros([len(classes), len(classes)+1]))

# 获取图片宽高
def get_image_width_high(full_image_name):
    image = Image.open(full_image_name)
    image_width, image_high = image.size[0], image.size[1]
    return image_width, image_high


# 读取原始标注数据
def read_label_txt(full_label_name, full_image_name):
    fp = open(full_label_name, mode="r")
    lines = fp.readlines()
    image_width, image_high = get_image_width_high(full_image_name)
    object_list = []
    for line in lines:
        array = line.split()
        x_label_min = (float(array[1]) - float(array[3]) / 2) * image_width
        x_label_max = (float(array[1]) + float(array[3]) / 2) * image_width
        y_label_min = (float(array[2]) - float(array[4]) / 2) * image_high
        y_label_max = (float(array[2]) + float(array[4]) / 2) * image_high
        bbox = [round(x_label_min, 2), round(y_label_min, 2), round(x_label_max, 2), round(y_label_max, 2)]
        category = int(array[0])
        obj_info = {
            'category' : category,
            'bbox' : bbox
        }
        object_list.append(obj_info)
    return object_list


# 计算交集面积
def label_area_detect(label_bbox_list, detect_bbox_list):
    x_label_min, y_label_min, x_label_max, y_label_max = label_bbox_list
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_bbox_list
    if (x_label_max <= x_detect_min or x_detect_max < x_label_min) or ( y_label_max <= y_detect_min or y_detect_max <= y_label_min):
        return 0
    else:
        lens = min(x_label_max, x_detect_max) - max(x_label_min, x_detect_min)
        wide = min(y_label_max, y_detect_max) - max(y_label_min, y_detect_min)
        return lens * wide

# 计算矩形框面积
def bbox_area(bbox_list):
    x_label_min, y_label_min, x_label_max, y_label_max = bbox_list
    return (x_label_max - x_label_min) * (y_label_max - y_label_min)


# label 匹配 detect
def label_match_detect(image_name, label_list, detect_list):
    for label in label_list:
        area_max = 0
        area_category = 0
        label_category = label['category']
        label_bbox = label['bbox']
        label_area = bbox_area(label_bbox)
        for detect in detect_list:
            if detect['name'] == image_name:
                detect_bbox = detect['bbox']
                detect_area = bbox_area(detect_bbox)
                area = label_area_detect(label_bbox, detect_bbox)
                if detect_area > (3*area) or label_area > (3*area):
                    continue
                if area > area_max:
                    area_max = area
                    area_category = detect['category']
        # 如果交集面积为0，矩阵最后一列+1
        if area_category == 0:
            result_list[int(label_category)][-1] += 1
        else:
            result_list[int(label_category)][classes.index(str(area_category))] += 1


def main():
    image_path = '../image_data/seed/test/images/'  # 图片文件路径
    label_path = '../image_data/seed/test/labels/'  # 标注文件路径
    detect_path = 'runs/detect/cbam/result.json'  # 预测的数据
    precision = 0     # 精确率
    recall = 0        # 召回率
    f1 = 0
    # 读取 预测 文件数据ss
    with open(detect_path, 'r') as load_f:
        detect_list = json.load(load_f)
    # 读取图片文件数据
    all_image = os.listdir(image_path)
    for i in range(len(all_image)):
        full_image_path = os.path.join(image_path, all_image[i])
        # 分离文件名和文件后缀
        image_name, image_extension = os.path.splitext(all_image[i])
        # 拼接标注路径
        full_label_path = os.path.join(label_path, image_name + '.txt')
        # 读取标注数据
        label_list = read_label_txt(full_label_path, full_image_path)
        # 标注数据匹配detect
        label_match_detect(all_image[i], label_list, detect_list)
    print(result_list)
    for i in range(len(classes)):
        row_sum, col_sum = sum(result_list[i]), sum(result_list[r][i] for r in range(len(classes)))
        precision_ = result_list[i][i] / float(col_sum)
        precision += precision_
        recall_ = result_list[i][i] / float(row_sum)
        recall += recall_
        f1_ = 2 * precision_ * recall_ / (precision_ + recall_)
        f1 += f1_
        print(f'{i}  precision: {precision_} recall: {recall_} f1:{f1_}')
    precision = precision / len(classes) * 100
    recall = recall / len(classes) * 100
    f1 = f1 / len(classes) * 100
    print(f'precision: {precision}%  recall: {recall}% '
          f'f1：{f1}%')




if __name__ == '__main__':
    main()

