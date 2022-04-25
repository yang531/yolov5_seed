# 多线程拓展数据集：1、调整亮度  2、镜像翻转

import os
import cv2 as cv
import random as random
import threading
import math
from shutil import copy2
import copy
from skimage import exposure

image_path = '../image_data/seed_10/seed_10_origin/images/'
label_path = '../image_data/seed_10/seed_10_origin/labels/'
image_save_path = '../image_data/seed_10/images_expansion/'
label_save_path = '../image_data/seed_10/labels_expansion/'
threed_nums = 10  # CPU线程数


# 改变图像亮度，label数据不变
def change_light(image_name, image):
    flag = random.uniform(0.5, 1.5)  #flag>1为调暗,小于1为调亮
    image_light = exposure.adjust_gamma(image, flag)
    save_images_labels(image_name, 'light', image_light, False)


# 镜像翻转，随机水平或者垂直翻转
def flip_pic_bboxes(image_name, image, bboxes):
    # ---------------------- 镜像图像 ----------------------
    flip_img = copy.deepcopy(image)
    if random.random() < 0.5:
        horizon = True
    else:
        horizon = False
    image_high, image_width, _ = image.shape
    if horizon:  # 水平翻转
        flip_img = cv.flip(flip_img, 1)
    else:
        flip_img = cv.flip(flip_img, 0)
    # ---------------------- 矫正boundingbox ----------------------
    flip_bboxes = list()
    for bbox in bboxes:
        name, x_min, y_min, x_max, y_max = bbox
        if horizon:
            x_min_flip = image_width - x_max
            y_min_flip = y_min
            x_max_flip = image_width - x_min
            y_max_flip = y_max
        else:
            x_min_flip = x_min
            y_min_flip = image_high - y_max
            x_max_flip = x_max
            y_max_flip = image_high - y_min
        # 计算中心点的 相对 x, y 坐标, w,h 的值
        x = (x_min_flip + x_max_flip) / 2 / image_width
        y = (y_min_flip + y_max_flip) / 2 / image_high
        w = (x_max_flip - x_min_flip) / image_width
        h = (y_max_flip - y_min_flip) / image_high
        flip_bboxes.append([name, x, y, w, h])

    save_images_labels(image_name, 'flip', flip_img, flip_bboxes)


# 保存images 和 labels
def save_images_labels(image_name, expansion_name, image_data, label_data):
    name, ext = os.path.splitext(image_name)  # 分离文件名和拓展名
    label_name = name + '.txt'
    image_save_name = name + '_' + expansion_name + ext
    label_save_name = name + '_' + expansion_name + '.txt'
    image_save_full_path = os.path.join(image_save_path, image_save_name)
    label_save_full_path = os.path.join(label_save_path, label_save_name)
    # 保存变化后的图片
    cv.imwrite(image_save_full_path, image_data)
    if label_data:
        fp = open(label_save_full_path, mode="w", encoding="utf-8")
        file_str = ''
        for bbox in label_data:
            file_str += str(bbox[0]) + ' ' + str(round(bbox[1], 6)) + ' ' + str(round(bbox[2], 6)) + \
                       ' ' + str(round(bbox[3], 6)) + ' ' + str(round(bbox[4], 6)) + '\n'
        fp.write(file_str.strip('\n'))
        fp.close()
    else:
        # label文件数据不变，直接复制
        copy2(os.path.join(label_path, label_name), label_save_full_path)


# 读取原始标注数据
def read_label_txt(full_label_name, image):
    fp = open(full_label_name, mode="r")
    lines = fp.readlines()
    image_high, image_width, _ = image.shape
    bboxes = []
    for line in lines:
        array = line.split()
        x_min = (float(array[1]) - float(array[3]) / 2) * image_width
        x_max = (float(array[1]) + float(array[3]) / 2) * image_width
        y_min = (float(array[2]) - float(array[4]) / 2) * image_high
        y_max = (float(array[2]) + float(array[4]) / 2) * image_high
        bbox = [array[0], round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)]
        bboxes.append(bbox)
    return bboxes



# 分割list数据
def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]


def main(files):
    for file in files:
        last_file = os.path.join(image_path, file)
        image = cv.imread(last_file)
        # 修改图像亮度
        change_light(file, image)
        # 获取图像标注数据
        name, ext = os.path.splitext(file)  # 分离文件名和拓展名
        full_label_name = os.path.join(label_path, name + '.txt')
        label_data = read_label_txt(full_label_name, image)
        # 图像镜像翻转
        flip_pic_bboxes(file, image, label_data)


if __name__ == '__main__':
    files = os.listdir(image_path)
    image_nums = math.ceil(len(files) / threed_nums)
    image_list = list_split(files, int(image_nums))
    for i in range(threed_nums):
        t = threading.Thread(target=main, args=(image_list[i],))
        t.start()
        print(f'threed {i} is running.')