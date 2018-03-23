# -*- coding: utf-8 -*-
import sys
import os

sys.path.append('./')

from model_hourglass import HourglassModelForClothes
from time import time, clock, sleep
import numpy as np
import tensorflow as tf
import scipy.io
from train_clothes import process_config_clothes
import cv2
from predictClothes import PredictClothes
from yolo_net import YOLONet
from datagenclothes import DataGenClothes
import config as cfg
from filters import VideoFilters
from itertools import islice
import csv
import numpy as np
from inferenceclothes import InferenceClothes


def predict_one_category(config_file, category, model='hg_clothes_001_200', ):
    """ predict all test image,write into result.csv
        Args:
            config_file : the model  training config
            category    : one of ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
            model       : the name using testing

        ['neckline_left0', 'neckline_right1', 'center_front2', 'shoulder_left3', 'shoulder_right4', 'armpit_left5',
     'armpit_right6', 'waistline_left7', 'waistline_right8', 'cuff_left_in9', 'cuff_left_out10', 'cuff_right_in11',
     'cuff_right_out12', 'top_hem_left13', 'top_hem_right14', 'waistband_left15', 'waistband_right16', 'hemline_left17',
     'hemline_right18', 'crotch19', 'bottom_left_in20', 'bottom_left_out21', 'bottom_right_in22', 'bottom_right_out23']
        """

    blouse = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, ]
    outwear = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    trousers = [15, 16, 19, 20, 21, 22, 23]
    skirt = [15, 16, 17, 18]
    dress = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]
    cat = category.pop()
    if cat not in ['blouse', 'dress', 'outwear', 'skirt', 'trousers']:
        raise ValueError('category not blouse,dress,outwear,skirt,trousers')
    # inf = InferenceClothes(config_file, model)
    params = process_config_clothes(config_file)
    print(params)
    csvresult = open('result' + cat + '.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvresult)
    starttime = time()
    # with open(params['test_csv_file'], "r") as f:
    with open('test_1.csv', "r") as f:
        for value in islice(f, 1, None):  # 读取去掉第一行之后的数据
            value = value.strip().split(',')
            img_name = value[0]
            img_category = value[1]
            if img_category == cat:
                try:
                    img = cv2.imread(os.path.join(params['img_test_dir'], img_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height = img.shape[0]
                    width = img.shape[1]
                    # print('height',height,'width',width)
                    img = cv2.resize(img, (256, 256))
                    # print(img.shape)
                    # predjoints = inf.predictJoints(img)
                    predjoints = np.arange(48).reshape((24, 2))
                    joints = []
                    joints.append(img_name)
                    joints.append(img_category)
                    for i in range(predjoints.shape[0]):
                        joints.append(
                            str(int(predjoints[i][1] / 256 * width)) + '_' + str(
                                int(predjoints[i][0] / 256 * height)) + '_1')
                    print(joints)
                    writer.writerow(joints)
                except:
                    print("Not find the image:", img_name)
    csvresult.close()
    print("test images in", time() - starttime, " sec")


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 2:
        c = argv[1]
    else:
        raise ValueError('need a parameter in b is blouse,d is dress,o is outwear,s is skirt,t is trousers\n'
                         'for example: python testonecategoryimage.py b')
    name = os.name
    if name == 'nt':
        config_file = 'config_clothes_win.cfg'
    else:
        config_file = 'config_clothes.cfg'
    params = process_config_clothes(config_file)
    category = []
    if c == 'b':
        category.append('blouse')
        cat = 'blouse'
    elif c == 'd':
        category.append('dress')
        cat = 'dress'
    elif c == 'o':
        category.append('outwear')
        cat = 'outwear'
    elif c == 's':
        category.append('skirt')
        cat = 'skirt'
    elif c == 't':
        category.append('trousers')
        cat = 'trousers'
    print('categoty =', category)
    model = params['name'] + cat
    predict_one_category(config_file, category, model)
