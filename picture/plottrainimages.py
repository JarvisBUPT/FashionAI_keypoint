import csv
import os
import sys
from time import time

import cv2
import numpy as np

sys.path.append('../')
from configs.processconfig import process_config_clothes


def plt_skeleton(img, joints, joints_name):
    """ Given an Image, returns Image with plotted limbs (TF VERSION)
    Args:
        img 	    : Original Image
        joints      : List the return of relative cordinate of thr original image
        joints_name : List, the name of joints, the length is the same of the joints
    """
    color = [(0, 0, 255), (196, 203, 128), (136, 150, 0), (64, 77, 0),
             (201, 230, 200), (132, 199, 129), (71, 160, 67), (32, 94, 27),
             (130, 224, 255), (7, 193, 255), (0, 160, 255), (0, 111, 255),
             (0, 255, 0), (174, 164, 144), (139, 125, 96), (100, 90, 69),
             (252, 229, 179), (247, 195, 79), (229, 155, 3), (155, 87, 1),
             (231, 190, 225), (200, 104, 186), (176, 39, 156), (162, 31, 123),
             (210, 205, 255), (115, 115, 229), (80, 83, 239), (40, 40, 198)]
    img_copy = np.copy(img)
    for i in range(0, len(joints), 2):
        cv2.circle(img, (joints[i], joints[i + 1]), 2, color[i // 2], 1)
        cv2.putText(img, str(i // 2) + joints_name[i // 2], (joints[i], joints[i + 1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color[i // 2], 1)
        cv2.putText(img, str(i // 2) + joints_name[i // 2], (10, 10 + i * 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color[i // 2], 1)
    return img


def main():
    with open('fashionAI_key_points_test_b_answer_20180426.csv', "r") as f:
        starttime = time()
        # with open('result_04131200.csv', "r") as f:
        print('first process_config_clothes', os.getcwd())
        params = process_config_clothes()
        print('last process_config_clothes', os.getcwd())
        print('params', params)
        reader = csv.reader(f)
        next(reader)
        imgs_dir_test = params['img_test_dir']
        print(imgs_dir_test)
        img_num = 0
        for i, value in enumerate(reader):  # 读取去掉第一行之后的数据
            img_num = img_num + 1
            img_name = value[0]
            cat_temp = value[1]
            # print(img_name, cat_temp)
            img = cv2.imread(os.path.join(imgs_dir_test, img_name))
            # print(img.shape)
            keypoint = value[2:]
            joints_plot = []
            for k in keypoint:
                joint = k.split('_')
                joints_plot.append(int(joint[0]))
                joints_plot.append(int(joint[1]))
            print(joints_plot)
            img_dst = os.path.join('trainplot', 'result_final_1',
                                   cat_temp)
            print(img_dst)
            if not os.path.exists(img_dst):
                os.makedirs(img_dst)
            img_plot = plt_skeleton(img, joints_plot, params['joint_list'])
            cv2.imwrite(os.path.join(img_dst, img_name.split('/')[2]), img_plot)
    print('plot %d images totals %.3f s' % (img_num, (time() - starttime)))


if __name__ == '__main__':
    main()
