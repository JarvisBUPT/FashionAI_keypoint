# -*- coding: utf-8 -*-
import csv
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as scm
from skimage import transform


class DataGenClothes(object):
    """ DataGenerator Class : To generate Train, Validatidation and Test sets
    for the Deep Human Pose Estimation Model

    Return:
        Outputs have a shape of (Number of Image) X (Number of Stacks) X (Heigth: 64) X (Width: 64) X (OutputDimendion:
        number of joints,13,14,15,4 or 7)
    Joints:
        We use the tianchi clothes keypoint convention on joints numbering
        List of joints:
            0--neckline_left
            1--neckline_right
            2--center_front
            3--shoulder_left
            4--shoulder_right
            5--armpit_left
            6--armpit_right
            7--waistline_left
            8--waistline_right
            9--cuff_left_in
            10--cuff_left_out
            11--cuff_right_in
            12--cuff_right_out
            13--top_hem_left
            14--top_hem_right
            15--waistband_left
            16--waistband_right
            17--hemline_left
            18--hemline_right
            19--crotch
            20--bottom_left_in
            21--bottom_left_out
            22--bottom_right_in
            23--bottom_right_out
    Note:self.data_dict the key is the image name (Images/blouse/00a2b0f3f13413cd87fa51bb4e25fdfd.jpg，
    the value is another dict,which keyis "joints"、"visible"、"visible"、"category".


   """

    def __init__(self, joints_list=None, img_dir=None, train_data_file=None, category=None, cat=None):
        """ Initializer
        Args:
            joints_list			: List of joints condsidered
            img_dir				: Directory containing clothes category
            train_data_file		: csv file with training set data
            category    		: List of clothes category
            cat                 : str for category of the image: blouse, dress, outwear, skirt, trousers, ''
        Instance Var:
            self.joints_list    : List of name of clothes Keypoint
            self.category       : List of clothes category
            self.images         : List of names of all images
            self.category       : List of clthes category
        """

        if joints_list is None:
            joints_list = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                           'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                           'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                           'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                           'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
        if img_dir is None:
            img_dir = '/home/sk39/workspace/dataset/tianchi_clothes/train/'
        if train_data_file is None:
            train_data_file = '/home/sk39/workspace/dataset/tianchi_clothes/train/Annotations/train.csv'
        if category is None:
            category = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
        if cat is None:
            cat = ''
        self.joints_list = joints_list
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.category = category
        self.cat = cat

        self.images = []
        img_dir_temp = os.path.join(img_dir, "Images")
        print(img_dir)
        for k in category:
            img_dir_cat = os.path.join(img_dir_temp, k)
            self.images.extend(os.listdir(img_dir_cat))
        # print(self.images)
        print(self.images.__len__())

    # --------------------Generator Initialization Methods ---------------------


    def _create_train_table(self):
        """ Create Table of samples from TEXT file
        """
        self.train_table = []  # 记录图片名字的list
        self.no_intel = []  # 记录没有任何关键点标志的图片name的list
        self.data_dict = {}  # 以每张图片的name为key，记录每张图片的dict,key为"joints"、"visible"、"visible"、"category"

        print('Reading Train Data')
        start_time = time.time()

        with open(self.train_data_file, "r") as f:
            reader = csv.reader(f)
            if self.cat == '':
                next(reader)  # 对于全类型一起训练时，原始训练的csv需要去掉第一行
            for value in reader:
                # print(value)
                joint_ = []  # 记录每张图的关节点坐标x1,y1,x2,y2...，并且对于不可见和不存在的点的坐标变成-1，-1
                if len(value) == 0:
                    break
                name = value[0]
                if not os.path.exists(os.path.join(self.img_dir, name)):
                    self.no_intel.append(name)
                    continue
                category = value[1]
                self.train_table.append(name)
                keypoints = list(value[2:])  # 只截取关键点部位的坐标，x_y_visible,
                box = list([-1, -1, -1, -1])
                isvisible = []  # 每个关键点的可见度
                # joints = [map(int, cord.split('_')) for cord in keypoints]
                for cord in keypoints:
                    x, y, visible = cord.split('_')
                    # if visible == '0':
                    #     joint_.append(-1)
                    #     joint_.append(-1)
                    # else:
                    joint_.append(int(x))
                    joint_.append(int(y))
                    isvisible.append(int(visible))
                else:
                    joints = np.reshape(joint_, (-1, 2))
                    w = [1] * joints.shape[0]
                    for i in range(joints.shape[0]):
                        if np.array_equal(joints[i], [-1, -1]):
                            w[i] = 0
                    self.data_dict[name] = {'category': category, 'box': box, 'joints': joints, 'weights': w,
                                            'visible': isvisible}
        print("data_dict totals :", len(self.data_dict))
        print("no_intel totals :", len(self.no_intel))
        print("_create_train_table %f  s" % (time.time() - start_time))

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """
        for i in range(self.data_dict[name]['joints'].shape[0]):
            if np.array_equal(self.data_dict[name]['joints'][i], [-1, -1]):
                return False
        return True

    def _give_batch_name(self, batch_size=16, set='train'):
        """ Returns a List of Samples
        Args:
            batch_size	: Number of sample wanted
            set			: Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/valid')
                break
        return list_file

    def _create_sets(self, validation_rate=0.1):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate		: Percentage of validation data (in [0,1], don't waste time use 0.1)
        """
        print('Start Create Set ...')
        sample = len(self.train_table)
        print("train_table have %d samples" % sample)
        valid_sample = int(sample * validation_rate)
        print("val_sample have %d samples" % valid_sample)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = []
        # preset = self.train_table[sample - valid_sample:]
        # for elem in preset:
        #     if self._complete_sample(elem):
        #         self.valid_set.append(elem)
        #     else:
        #         self.train_set.append(elem)
        self.valid_set = self.train_table[sample - valid_sample:]
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')
        print('Set Created')

    def generateSet(self, rand=False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        if rand:
            self._randomize()
        self._create_sets()

        # ---------------------------- Generating Methods --------------------------

    def _makeGaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_hm(self, height, width, joints, maxlength, weight):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlength		: Length of the Bounding Box
        Returns:
            hm              : Dim is (H, W, num_joints)
        """
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints), dtype=np.float32)
        for i in range(num_joints):
            if not (np.array_equal(joints[i], [-1, -1])) and weight[i] == 1:
                s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2
                hm[:, :, i] = self._makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def _crop_data(self, height, width, box, joints, boxp=0.05):
        """ Automatically returns a padding vector and a bounding box given
        the size of the image and a list of joints.
        Args:
            height		: Original Height
            width		: Original Width
            box			: Bounding Box
            joints		: np array,is the coordinate (x, y) of joints, dim is (self.joints, 2),
                if the joint is invisible, let x = -1 and y = -1
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
        width_max, height_max = max(j[:, 0]), max(j[:, 1])
        j[joints == -1] = 1e5
        box[0], box[1] = min(j[:, 0]), min(j[:, 1])
        width_min, height_min = min(j[:, 0]), min(j[:, 1])
        # print('box', box)
        # crop_box (width_center,
        width_len = width_max - width_min  # 记录宽的长度
        heigth_len = height_max - height_min  # 记录高的长度
        # 将框扩大boxp
        width_min = width_min - int(boxp * width_len)
        height_min = height_min - int(boxp * heigth_len)
        width_max = width_max + int(boxp * width_len)
        height_max = height_max + int(boxp * heigth_len)
        # 处理扩大后框超过图片原始宽高的情况
        if width_min < 0: width_min = 0
        if height_min < 0: height_min = 0
        if width_max > width - 1: width_max = width - 1
        if height_max > height - 1: height_max = height - 1
        new_h = int(height_max - height_min)
        new_w = int(width_max - width_min)
        crop_box = [width_min + new_w // 2, height_min + new_h // 2, new_w, new_h]
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

    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	    : Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        max_lenght = max(crop_box[2], crop_box[3])
        # if max_lenght < 256:
        #     max_lenght = 256
        img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
              crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        # print('crop img shape', img.shape)
        # cv2.imwrite('ce.jpg', img)
        # width = img[]
        return img

    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Image
            hm			: Source Heat Map
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        hm = np.pad(hm, padding, mode='constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
              crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        hm = hm[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
             crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        return img, hm

    def _relative_joints(self, box, padding, joints, to_size=64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        将原始坐标归一化到64范围内
        Args:
            box		: Bounding Box
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        max_l = max(box[2], box[3])
        # 将以前坐标系中的（0-pad[1][0],0-pad[0][0])点变成新的原点建立新的坐标系，所以新的关键点坐标都加上pad
        new_j = new_j + [padding[1][0], padding[0][0]]
        # 让关键点的坐标剪切原始框扩展后的正方形框的左上角对应的x，y坐标。即以框的左上角为原点坐标。
        new_j = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
        # 将正方形框归一化为大小为64的新框下的坐标
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j.astype(np.int32)

    def _augment(self, img, hm, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            img = transform.rotate(img, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
        return img, hm

        # ----------------------- Batch Generator ----------------------------------

    def _generator(self, batch_size=16, stacks=4, set='train', stored=False, normalize=True, debug=False):
        """ Create Generator for Training
        Args:
            batch_size	    : Number of images per batch
            stacks			: Number of stacks/module in the network
            set				: Training/Testing/Validation set # TODO: Not implemented yet
            stored			: Use stored Value # TODO: Not implemented yet
            normalize		: True to return Image Value between 0 and 1
            _debug			: Boolean to test the computation time (/!\ Keep False)
        # Done : Optimize Computation time
            16 Images --> 1.3 sec (on i7 6700hq)
        """
        while True:
            if debug:
                t = time.time()
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
            files = self._give_batch_name(batch_size=batch_size, set=set)
            for i, name in enumerate(files):
                if name[:-1] in self.images:
                    try:
                        img = self.open_img(name)
                        joints = self.data_dict[name]['joints']
                        box = self.data_dict[name]['box']
                        weight = self.data_dict[name]['weights']
                        category = self.data_dict[name]['category']
                        visible = self.data_dict[name]['visible']
                        if debug:
                            print(box)
                        padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
                        if debug:
                            print(cbox)
                            print('maxl :', max(cbox[2], cbox[3]))
                        new_j = self._relative_joints(cbox, padd, joints, to_size=64)
                        hm = self._generate_hm(64, 64, new_j, 64, weight)
                        img = self._crop_img(img, padd, cbox)
                        img = img.astype(np.uint8)
                        # On 16 image per batch
                        # Avg Time -OpenCV : 1.0 s -skimage: 1.25 s -scipy.misc.imresize: 1.05s
                        img = scm.imresize(img, (256, 256))
                        # Less efficient that OpenCV resize method
                        # img = transform.resize(img, (256,256), preserve_range = True, mode = 'constant')
                        # May Cause trouble, bug in OpenCV imgwrap.cpp:3229
                        # error: (-215) ssize.area() > 0 in function cv::resize
                        # img = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)
                        img, hm = self._augment(img, hm)
                        hm = np.expand_dims(hm, axis=0)  # Dim is (1, 64, 64, 24)
                        hm = np.repeat(hm, stacks, axis=0)  # Dim is (4, 64. 64. 24)
                        if normalize:
                            train_img[i] = img.astype(np.float32) / 255
                        else:
                            train_img[i] = img.astype(np.float32)
                        train_gtmap[i] = hm
                    except:
                        i = i - 1
                else:
                    i = i - 1
            if debug:
                print('Batch : ', time.time() - t, ' sec.')
            yield train_img, train_gtmap

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
            train_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
            i = 0
            while i < batch_size:
                if sample_set == 'train':
                    name = random.choice(self.train_set)
                elif sample_set == 'valid':
                    name = random.choice(self.valid_set)
                joints = self.data_dict[name]['joints']
                box = self.data_dict[name]['box']
                weight = np.asarray(self.data_dict[name]['weights'])
                category = self.data_dict[name]['category']
                visible = self.data_dict[name]['visible']
                train_weights[i] = weight
                img = self.open_img(name)
                padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
                new_j = self._relative_joints(cbox, padd, joints, to_size=64)
                hm = self._generate_hm(64, 64, new_j, 64, weight)
                # TODO: save the heatmap
                '''
                hm_path = os.path.join('hourglass_saver', 'heatmap')
                if not os.path.exists(hm_path):
                    os.makedirs(hm_path)
                cv2.imwrite(os.path.join(hm_path, name.split('/')[2]), hm)
                '''
                img = self._crop_img(img, padd, cbox)
                img = img.astype(np.uint8)
                img = scm.imresize(img, (256, 256))
                # img = cv2.resize(img, (256, 256))
                img, hm = self._augment(img, hm)
                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, stacks, axis=0)
                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                else:
                    train_img[i] = img.astype(np.float32)
                train_gtmap[i] = hm
                i = i + 1
            yield train_img, train_gtmap, train_weights

    def generator(self, batchSize=16, stacks=4, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)

        # ---------------------------- Image Reader --------------------------------

    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample Images/blouse/155ee7793d159e227afb5f2e87ecf37b.jpg
            color	: Color Mode (RGB/BGR/GRAY)
        """
        # print(os.path.join(self.img_dir, name))
        img = cv2.imread(os.path.join(self.img_dir, name))
        # if color == 'RGB':
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     return img
        # elif color == 'BGR':
        #     return img
        # elif color == 'GRAY':
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # else:
        #     print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')
        return img

    def plot_img(self, name, plot='cv2'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()

    def test(self, toWait=0.2):
        """ TESTING METHOD
        You can run it to see if the preprocessing is well done.
        Wait few seconds for loading, then diaporama appears with image and highlighted joints
        /!\ Use Esc to quit
        Args:
            toWait : In sec, time between pictures
        """
        self._create_train_table()
        self._create_sets()
        for i in range(len(self.train_set)):
            img = self.open_img(self.train_set[i])
            w = self.data_dict[self.train_set[i]]['weights']
            padd, box = self._crop_data(img.shape[0], img.shape[1], self.data_dict[self.train_set[i]]['box'],
                                        self.data_dict[self.train_set[i]]['joints'], boxp=0.0)
            new_j = self._relative_joints(box, padd, self.data_dict[self.train_set[i]]['joints'], to_size=256)
            rhm = self._generate_hm(256, 256, new_j, 256, w)
            rimg = self._crop_img(img, padd, box)
            # See Error in self._generator
            # rimg = cv2.resize(rimg, (256,256))
            rimg = scm.imresize(rimg, (256, 256))
            # rhm = np.zeros((256,256,16))
            # for i in range(16):
            #	rhm[:,:,i] = cv2.resize(rHM[:,:,i], (256,256))
            grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', grimg / 255 + np.sum(rhm, axis=2))
            # Wait
            time.sleep(toWait)
            if cv2.waitKey(1) == 27:
                print('Ended')
                cv2.destroyAllWindows()
                break

                # ------------------------------- PCK METHODS-------------------------------

    def pck_ready(self, idlh=3, idrs=12, testSet=None):
        """ Creates a list with all PCK ready samples
        (PCK: Percentage of Correct Keypoints)
        """
        id_lhip = idlh
        id_rsho = idrs
        self.total_joints = 0
        self.pck_samples = []
        for s in self.data_dict.keys():
            if testSet == None:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
            else:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][
                    id_rsho] == 1 and s in testSet:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
        print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.Joints', self.total_joints)

    def getSample(self, sample=None):
        """ Returns information of a sample
        Args:
            sample : (str) Name of the sample
        Returns:
            img: RGB Image
            new_j: Resized Joints
            w: Weights of Joints
            joint_full: Raw Joints
            max_l: Maximum Size of Input Image
        """
        if sample != None:
            try:
                joints = self.data_dict[sample]['joints']
                box = self.data_dict[sample]['box']
                w = self.data_dict[sample]['weights']
                category = self.data_dict[sample]['category']
                visible = self.data_dict[sample]['visible']
                img = self.open_img(sample)
                padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.1)
                new_j = self._relative_joints(cbox, padd, joints, to_size=256)
                joint_full = np.copy(joints)
                max_l = max(cbox[2], cbox[3])
                joint_full = joint_full + [padd[1][0], padd[0][0]]
                joint_full = joint_full - [cbox[0] - max_l // 2, cbox[1] - max_l // 2]
                img = self._crop_img(img, padd, cbox)
                img = img.astype(np.uint8)
                img = scm.imresize(img, (256, 256))
                return img, new_j, w, joint_full, max_l
            except:
                return False
        else:
            print('Specify a sample name')


if __name__ == '__main__':
    from configs.processconfig import process_config_clothes

    print('--Parsing Config File')
    params = process_config_clothes()
    print(params)
    dataset = DataGenClothes(params['joint_list'], params['img_directory'], params['training_txt_file'],
                             params['category'])
    # dataset = DataGenClothes(params['joint_list'], params['img_directory'], 'train_1.csv',
    #                          params['category'])
    dataset._create_train_table()
    dataset._randomize()
    dataset._create_sets()
    img_name = "Images/skirt/00ef62ab642415f214abbc675f28d5e9.jpg"
    img_name_win = r"Images\skirt\00ef62ab642415f214abbc675f28d5e9.jpg"
    img = dataset.open_img(img_name_win)
    print(img.shape)
    box = [-1, -1, -1, -1]
    padd, cbox = dataset._crop_data(img.shape[0], img.shape[1], box, dataset.data_dict[img_name]['joints'], boxp=0.1)
    dataset._crop_img(img, padd, cbox)
    # new_j = dataset._relative_joints(cbox, padd, dataset.joints, to_size=64)
    # hm = dataset._generate_hm(64, 64, new_j, 64, dataset.weight)
    # img = dataset._crop_img(img, padd, cbox)
    # img = img.astype(np.uint8)
    # img = scm.imresize(img, (256, 256))
    # # img = cv2.resize(img, (256, 256))
    # img, hm = dataset._augment(img, hm)
    # hm = np.expand_dims(hm, axis=0)
    # hm = np.repeat(hm, 4, axis=0)

    # i = 0
    # for k in dataset._aux_generator():
    #     i += 1
    #     print("...")
    #     if i == 1: break
    # # padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
    print("end")
