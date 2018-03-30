import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
import csv
from processconfig import process_config_clothes
from itertools import islice

if __name__ == '__main__':
    params = process_config_clothes()
    category = params['category']
    blouse = params['blouse']
    outwear = params['outwear']
    trousers = params['trousers']
    skirt = params['skirt']
    dress = params['dress']

    print(dress)
    fblouse = open("split_blouse.csv", 'w', newline='')
    fdress = open("split_dress.csv", 'w', newline='')
    foutwear = open("split_outwear.csv", 'w', newline='')
    fskirt = open("split_skirt.csv", 'w', newline='')
    ftrousers = open("split_trousers.csv", 'w', newline='')

    writerblouse = csv.writer(fblouse)
    writerdress = csv.writer(fdress)
    writeroutwear = csv.writer(foutwear)
    writerskirt = csv.writer(fskirt)
    writertrousers = csv.writer(ftrousers)

    with open(params['training_txt_file'], "r") as f:
        for value in islice(f, 1, None):  # 读取去掉第一行之后的数据
            joint = []  # 存储值为x1_y1_visible
            joint_ = []  # 记录每张图的关节点坐标x1,y1,x2,y2...，并且对于不可见和不存在的点的坐标变成-1，-1
            value = value.strip()
            if value == '':
                break
            line = value.split(',')
            name = line[0]
            cat = line[1]
            joint.append(name)
            joint.append(cat)
            joint_.append(name)
            joint_.append(cat)
            keypoints = list(line[2:])  # 只截取关键点部位的坐标，x_y_visible,
            print(keypoints)
            # if cat == 'blouse':
            for i, cord in enumerate(keypoints):
                if i in blouse:
                    joint.append(cord)
                    x, y, visible = cord.split('_')
                    if visible == '0':
                        joint_.append(-1)
                        joint_.append(-1)
                    else:
                        joint_.append(int(x))
                        joint_.append(int(y))
            writerblouse.writerow(joint)
                # writerblouse.writerow(joint_)
            # if cat == 'dress':
            for i, cord in enumerate(keypoints):
                if i in dress:
                    joint.append(cord)
                    x, y, visible = cord.split('_')
                    if visible == '0':
                        joint_.append(-1)
                        joint_.append(-1)
                    else:
                        joint_.append(int(x))
                        joint_.append(int(y))
            writerdress.writerow(joint)
            # writerdress.writerow(joint_)
            # if cat == 'outwear':
            for i, cord in enumerate(keypoints):
                if i in outwear:
                    joint.append(cord)
                    x, y, visible = cord.split('_')
                    if visible == '0':
                        joint_.append(-1)
                        joint_.append(-1)
                    else:
                        joint_.append(int(x))
                        joint_.append(int(y))
            writeroutwear.writerow(joint)
                # writeroutwear.writerow(joint_)
            # if cat == 'skirt':
            for i, cord in enumerate(keypoints):
                if i in skirt:
                    joint.append(cord)
                    x, y, visible = cord.split('_')
                    if visible == '0':
                        joint_.append(-1)
                        joint_.append(-1)
                    else:
                        joint_.append(int(x))
                        joint_.append(int(y))
            writerskirt.writerow(joint)
                # writerskirt.writerow(joint_)
            # if cat == 'trousers':
            for i, cord in enumerate(keypoints):
                if i in trousers:
                    joint.append(cord)
                    x, y, visible = cord.split('_')
                    if visible == '0':
                        joint_.append(-1)
                        joint_.append(-1)
                    else:
                        joint_.append(int(x))
                        joint_.append(int(y))
            writertrousers.writerow(joint)
                # writertrousers.writerow(joint_)
    fblouse.close()
    fdress.close()
    foutwear.close()
    fskirt.close()
    ftrousers.close()
