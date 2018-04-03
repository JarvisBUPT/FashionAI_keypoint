import sys
import os

sys.path.append('./')

import numpy as np
from time import time, clock
from processconfig import process_config_clothes
import cv2
from inferenceclothes import InferenceClothes
from datagenclothes import DataGenClothes

path_coat = '/home/sk39/workspace/cheng/clothes_tags/data/coat_length_labels/train/'
save_coat = '/home/sk39/workspace/cheng/clothes_tags/data2/coat_length_labels/train/'
path_lapel = '/home/sk39/workspace/cheng/clothes_tags/data/lapel_design_labels/train/'
save_lapel = '/home/sk39/workspace/cheng/clothes_tags/data2/lapel_design_labels/train/'
path_neckline = '/home/sk39/workspace/cheng/clothes_tags/data/neckline_design_labels/train'
save_neckline = '/home/sk39/workspace/cheng/clothes_tags/data2/neckline_design_labels/train'
path_skirt = '/home/sk39/workspace/cheng/clothes_tags/data/skirt_length_labels/train'
save_skirt = '/home/sk39/workspace/cheng/clothes_tags/data2/skirt_length_labels/train'
path_collar = '/home/sk39/workspace/cheng/clothes_tags/data/collar_design_labels/train'
save_collar = '/home/sk39/workspace/cheng/clothes_tags/data2/collar_design_labels/train'
path_neck = '/home/sk39/workspace/cheng/clothes_tags/data/neck_design_labels/train'
save_neck = '/home/sk39/workspace/cheng/clothes_tags/data2/neck_design_labels/train'
path_pant = '/home/sk39/workspace/cheng/clothes_tags/data/pant_length_labels/train'
save_pant = '/home/sk39/workspace/cheng/clothes_tags/data2/pant_length_labels/train'
path_sleeve = '/home/sk39/workspace/cheng/clothes_tags/data/sleeve_length_labels/train'
save_sleeve = '/home/sk39/workspace/cheng/clothes_tags/data2/sleeve_length_labels/train'

coat_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
collar_classes = ['0', '1', '2', '3', '4']
lapel_classes = ['0', '1', '2', '3', '4']
neck_classes = ['0', '1', '2', '3', '4']
neckline_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
pant_classes = ['0', '1', '2', '3', '4', '5']
skirt_classes = ['0', '1', '2', '3', '4', '5']
sleeve_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8']


def crop_data(height, width, box, joints, boxp=0.05):
    """ Automatically returns a padding vector and a bounding box given
    the size of the image and a list of joints.
    Args:
        height		: Original Height
        width		: Original Width
        box			: Bounding Box
        joints		: Array of joints
        boxp		: Box percentage (Use 20% to get a good bounding box)
    """
    # 图片以左上角为（0,0）点，往左为x轴，往下为y轴
    # 由于img.shape返回值（1280,720,3）第一个参数1280代表高，第二个参数720代表宽，和平常理解的顺序相反，3表示RGB三种通道
    # 所以padding[0][0]将会在原始图片的上边添加0，padding[1][0]会在图片左边加0。
    # padding = [[0, 0], [0, 0], [0, 0]]
    # crop_box = [width // 2, height // 2, width, height]
    padding = [[0, 0], [0, 0], [0, 0]]
    j = np.copy(joints)
    box[2], box[3] = max(j[:, 0]), max(j[:, 1])
    j[j < 0] = 1e5
    print(j)
    box[0], box[1] = min(j[:, 0]), min(j[:, 1])
    crop_box = [box[0] - int(boxp * (box[2] - box[0])), box[1] - int(boxp * (box[3] - box[1])),
                box[2] + int(boxp * (box[2] - box[0])), box[3] + int(boxp * (box[3] - box[1]))]
    if crop_box[0] < 0: crop_box[0] = 0
    if crop_box[1] < 0: crop_box[1] = 0
    if crop_box[2] > width - 1: crop_box[2] = width - 1
    if crop_box[3] > height - 1: crop_box[3] = height - 1
    new_h = int(crop_box[3] - crop_box[1])
    new_w = int(crop_box[2] - crop_box[0])
    crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
    if new_h > new_w:
        # bounds是为了防止以框的中心为原点，以max(new_h, new_w)为直径画框时超出图片的大小，超出的部分即为pad，通过补0完成
        bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
        if bounds[0] < 0:
            padding[1][0] = abs(bounds[0])
        if bounds[1] > width - 1:
            padding[1][1] = abs(width - bounds[1])
    elif new_h < new_w:
        bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
        if bounds[0] < 0:
            padding[0][0] = abs(bounds[0])
        if bounds[1] > width - 1:
            padding[0][1] = abs(height - bounds[1])
    # 将框的中心左边加上padding[1][0]是因为_crop_img函数不是以（0,0）点为原点，因为图片加了pad之后原点可能会变
    crop_box[0] += padding[1][0]
    crop_box[1] += padding[0][0]
    return padding, crop_box


def crop_img(img, padding, crop_box):
    """ Given a bounding box and padding values return cropped image
    Args:
        img			: Source Image
        padding	    : Padding
        crop_box	: Bounding Box
    """
    img = np.pad(img, padding, mode='constant')
    max_lenght = max(crop_box[2], crop_box[3])
    img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
          crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
    return img


def predictallimage(params, model='hg_clothes_001_199'):
    """ predict all test image,write into result.csv
    Args:
        params:
    """
    classes_all = [coat_classes, lapel_classes, neckline_classes, skirt_classes, collar_classes, neck_classes,
                   pant_classes, sleeve_classes]
    path_all = [path_coat, path_lapel, path_neckline, path_skirt, path_collar, path_neck, path_pant, path_sleeve]
    save_all = [save_coat, save_lapel, save_neckline, save_skirt, save_collar, save_neck, save_pant, save_sleeve]
    inf = InferenceClothes(params, model)
    dataset = DataGenClothes(params, img_dir=params['img_directory'])
    for i, cls in enumerate(classes_all):
        img_dir_temp = path_all[i]
        img_dir_save = save_all[i]
        print("img_dir_temp", img_dir_temp)
        for j in cls:
            print(cls, j)
            img_dir = os.path.join(img_dir_temp, j)
            img_dir_s = os.path.join(img_dir_save, j)
            print(img_dir_s)
            if not os.path.exists(img_dir_s):
                os.makedirs(img_dir_s)
            images = []
            images = os.listdir(img_dir)
            for img_name in images:
                # try:
                print("img_name", img_name)
                img_src = cv2.imread(os.path.join(img_dir, img_name))
                # img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
                height = img_src.shape[0]
                width = img_src.shape[1]
                print('height', height, 'width', width)
                img_input = cv2.resize(img_src, (256, 256))
                print(img_input.shape)
                predjoints = inf.predictJoints(img_input)
                # predjoints = np.arange(48).reshape((24, 2))
                joint_list = []
                for i in range(predjoints.shape[0]):
                    joint_list.append(int(predjoints[i][1] / 256 * width))
                    joint_list.append(int(predjoints[i][0] / 256 * height))
                joints = np.array(joint_list)
                joints = joints.reshape((-1, 2))
                # print(joints)
                box = [-1, -1, -1, -1]
                padd, cbox = crop_data(img_src.shape[0], img_src.shape[1], box, joints, boxp=0.2)
                print('padd:', padd, ' cbox:', cbox)
                img_crop = crop_img(img_src, padd, cbox)
                print(img_crop.shape)
                cv2.imwrite(os.path.join(img_dir_s, img_name), img_crop)

                # except:
                #     print("Not find the image:", img_name)


if __name__ == '__main__':
    params = process_config_clothes()
    predictallimage(params)
