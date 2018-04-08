# -*- coding: utf-8 -*-
import sys
import os

sys.path.append('./')

from model_hourglass import HourglassModelForClothes
from time import time, clock, sleep
import numpy as np
import tensorflow as tf
import scipy.io
from processconfig import process_config_clothes
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


def predict_one_category(params, category, model='hg_clothes_001_199', ):
    """ predict all test image,write into result.csv
        Args:
            params       : the model  needed parameters
            category    : List, one of ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
            model       : the name using testing

        """

    blouse = params['blouse']
    outwear = params['outwear']
    trousers = params['trousers']
    skirt = params['skirt']
    dress = params['dress']
    cat = category.pop()
    if cat not in ['blouse', 'dress', 'outwear', 'skirt', 'trousers']:
        raise ValueError('category not blouse,dress,outwear,skirt,trousers')
    name = params['name'] + cat  # params['name']=hg_clothes_001+'blouse'
    if cat == '':
        num_joints = 24
    else:
        num_joints = len(params[cat])
    joint_list = params['joint_list']
    joints = []  # 记录该类别对应的24个位置中的哪些值 如：[ 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left']
    if cat == '':
        joints = joint_list
    else:
        for i, v in enumerate(joint_list):
            if i in params[cat]:
                joints.append(v)
    print(joints)
    params['name'] = name
    params['num_joints'] = num_joints
    params['joint_list'] = joints
    print("test params", params)
    inf = InferenceClothes(params, model)
    pre = PredictClothes(params)
    print("Start predicting ...")
    csvresult = open('result' + cat + '.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvresult)
    starttime = time()
    with open(params['test_csv_file'], "r") as f:
        # with open('test_1.csv', "r") as f:
        for value in islice(f, 1, None):  # 读取去掉第一行之后的数据
            value = value.strip().split(',')
            img_name = value[0]
            cat_temp = value[1]
            print(img_name, cat_temp)
            if cat_temp == cat:
                img_path = os.path.join(params['img_test_dir'], img_name)
                img = cv2.imread(img_path)
                height = img.shape[0]
                width = img.shape[1]
                img_resize = cv2.resize(img, (256, 256))
                predjoints = inf.predictJoints(img_resize)
                # predjoints = np.arange(48).reshape((24, 2))
                joints = []  # 暂存需要提交结果的一行数据
                joints_plot = []  # 传递给plt_skeleton可视化输出结果
                joints.append(img_name)
                joints.append(cat_temp)
                # 将预测的每个关键点的xy坐标提取出来转换成实际图片的真实坐标
                for i in range(24):  # 一共24个点
                    if i in params[cat]:  # 这个点是否属于这个类
                        # self.params[cat] is like joints dress = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]
                        for j, v in enumerate(params[cat]):  # j 是下标， v是对应下标的值
                            if i == v:  # 获取i点在params[cat]中的下标j，该下标j对应predjoints的下标
                                joints.append(
                                    str(int(predjoints[j][1] / 256 * width)) + '_' + str(
                                        int(predjoints[j][0] / 256 * height)) + '_1')
                                joints_plot.append(int(predjoints[j][1] / 256 * width))
                                joints_plot.append(int(predjoints[j][0] / 256 * height))
                    else:
                        joints.append('-1_-1_-1')
                writer.writerow(joints)
                img_dst = os.path.join(params['log_dir_test'], cat_temp)
                # cat_temp, img_name.split('/')[2]
                if not os.path.exists(img_dst):
                    os.makedirs(img_dst)
                img_plot = pre.plt_skeleton(img, joints_plot, params['joint_list'])
                cv2.imwrite(os.path.join(img_dst, img_name.split('/')[2]), img_plot)
    csvresult.close()
    print("test images in", time() - starttime, " sec")


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 3:
        c = argv[1]
        epoch = argv[2]
    else:
        raise ValueError('need two parameter or one.One is in b is blouse,d is dress,o is outwear,'
                         's is skirt,t is trousers. Two is the number of epoch\n'
                         'for example: python testonecategoryimage.py b 101 or python testonecategoryimage.py b ')
    params = process_config_clothes()
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
    elif c == 'a':
        category = params['category']
        cat = ''
    print('categoty =', category)
    model = './hourglass_saver/model/' + params['name'] + cat + '/' + params['name'] + cat + "_" + epoch
    print('model name', model)
    # model = params['name'] + "_" + epoch
    predict_one_category(params, category, model)
