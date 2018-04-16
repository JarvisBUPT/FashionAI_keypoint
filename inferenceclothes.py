# -*- coding: utf-8 -*-
import os
import sys

sys.path.append('./')

from time import time, strftime
from configs.processconfig import process_config_clothes
import cv2
from predictClothes import PredictClothes
from video.filters import VideoFilters
from itertools import islice
import csv
import numpy as np


class InferenceClothes():
    """ Inference Class
    Use this file to make your prediction
    Easy to Use
    Images used for inference should be RGB images (int values in [0,255])
    Methods:
        webcamSingle    : Single Person Pose Estimation on Webcam Stream
        webcamMultiple  : Multiple Person Pose Estimation on Webcam Stream
        webcamPCA       : Single Person Pose Estimation with reconstruction error (PCA)
        webcamYOLO      : Object Detector
        predictHM       : Returns Heat Map for an input RGB Image
        predictJoints   : Returns joint's location (for a 256x256 image)
        pltSkeleton     : Plot skeleton on image
        runVideoFilter  : SURPRISE !!!
    """

    def __init__(self, params, model='hg_refined_tiny_200', yoloModel='YOLO_small.ckpt'):
        """ Initilize the Predictor
        Args:
            config_file 	 	: *.cfg file with model's parameters
            model 	 	 	 	: *.index file's name. (weights to load)
            yoloModel 	 	    : *.ckpt file (YOLO weights to load)
        """
        t = time()
        self.predict = PredictClothes(params)
        self.predict.color_palette()
        self.predict.links_joints()
        self.predict.model_init()
        self.predict.load_model(load=model)
        self.predict.yolo_init()
        self.predict.restore_yolo(load=yoloModel)
        self.predict._create_prediction_tensor()
        self.filter = VideoFilters()
        print('Done: ', time() - t, ' sec.')

    # -------------------------- WebCam Inference-------------------------------
    def webcamSingle(self, thresh=0.2, pltJ=True, pltL=True):
        """ Run Single Pose Estimation on Webcam Stream
        Args :
            thresh  : Joint Threshold
            pltJ    : (bool) True to plot joints
            pltL    : (bool) True to plot limbs
        """
        self.predict.hpeWebcam(thresh=thresh, plt_j=pltJ, plt_l=pltL, plt_hm=False, debug=False)

    def webcamMultiple(self, thresh=0.2, nms=0.5, resolution=800, pltL=True, pltJ=True, pltB=True, isolate=False):
        """ Run Multiple Pose Estimation on Webcam Stream
        Args:
            thresh      : Joint Threshold
            nms         : Non Maxima Suppression Threshold
            resolution  : Stream Resolution
            pltJ        : (bool) True to plot joints
            pltL        : (bool) True to plot limbs
            pltB        : (bool) True to plot bounding boxes
            isolate     : (bool) True to show isolated skeletons
        """
        self.predict.mpe(j_thresh=thresh, nms_thresh=nms, plt_l=pltL, plt_j=pltJ, plt_b=pltB, img_size=resolution,
                         skeleton=isolate)

    def webcamPCA(self, n=5, matrix='p4frames.mat'):
        """ Run Single Pose Estimation with Error Reconstruction on Webcam Stream
        Args:
            n       : Number of dimension to keep before reconstruction
            matrix  : MATLAB eigenvector matrix to load
        """
        self.predict.reconstructACPVideo(load=matrix, n=n)

    def webcamYOLO(self):
        """ Run Object Detection on Webcam Stream
        """
        cam = cv2.VideoCapture(0)
        return self.predict.camera_detector(cam, wait=0, mirror=True)

    # ----------------------- Heat Map Prediction ------------------------------

    def predictHM(self, img):
        """ Return Sigmoid Prediction Heat Map
        Args:
            img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
        """
        return self.predict.pred(self, img / 255, debug=False, sess=None)

    # ------------------------- Joint Prediction -------------------------------

    def predictJoints(self, img, mode='gpu', thresh=0.2):
        """ Return Joint Location
        /!\ Location with respect to 256x256 image
        Args:
            img     : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
            mode    : 'cpu' / 'gpu' Select a mode to compute joints' location
            thresh  : Joint Threshold
        """
        SIZE = False
        if len(img.shape) == 3:
            batch = np.expand_dims(img, axis=0)
            SIZE = True
        elif len(img.shape) == 4:
            batch = np.copy(img)
            SIZE = True
        if SIZE:
            if mode == 'cpu':
                return self.predict.joints_pred_numpy(batch / 255, coord='img', thresh=thresh, sess=None)
            elif mode == 'gpu':
                return self.predict.joints_pred(batch / 255, coord='img', debug=False, sess=None)
            else:
                print("Error : Mode should be 'cpu'/'gpu'")
        else:
            print('Error : Input is not a RGB image nor a batch of RGB images')

    # ----------------------------- Plot Skeleton ------------------------------

    def pltSkeleton(self, img, thresh, pltJ, pltL):
        """ Return an image with plotted joints and limbs
        Args:
            img     : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
            thresh  : Joint Threshold
            pltJ    : (bool) True to plot joints
            pltL    : (bool) True to plot limbs
        """
        return self.predict.pltSkeleton(img, thresh=thresh, pltJ=pltJ, pltL=pltL, tocopy=True, norm=True)

    # -------------------------- Video Processing ------------------------------

    def processVideo(self, source=None, outfile=None, thresh=0.2, nms=0.5, codec='DIVX', pltJ=True, pltL=True,
                     pltB=True, show=False):
        """ Run Multiple Pose Estimation on Video Footage
        Args:
            source      : Input Footage
            outfile     : File to Save
            thesh       : Joints Threshold
            nms         : Non Maxima Suppression Threshold
            codec       : Codec to use
            pltJ        : (bool) True to plot joints
            pltL        : (bool) True to plot limbs
            pltB        : (bool) True to plot bounding boxes
            show        : (bool) Show footage during processing
        """
        return self.predict.videoDetection(src=source, outName=outfile, codec=codec, j_thresh=thresh, nms_thresh=nms,
                                           show=show, plt_j=pltJ, plt_l=pltL, plt_b=pltB)

    # -------------------------- Process Stream --------------------------------

    def centerStream(self, img):
        img = cv2.flip(img, 1)
        img[:,
        self.predict.cam_res[1] // 2 - self.predict.cam_res[0] // 2:self.predict.cam_res[1] // 2 + self.predict.cam_res[
                                                                                                       0] // 2]
        img_hg = cv2.resize(img, (256, 256))
        img_res = cv2.resize(img, (800, 800))
        img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
        return img_res, img_hg

    def plotLimbs(self, img_res, j):
        """
        """
        for i in range(len(self.predict.links)):
            l = self.predict.links[i]['link']
            good_link = True
            for p in l:
                if np.array_equal(j[p], [-1, -1]):
                    good_link = False
            if good_link:
                pos = self.predict.givePixel(l, j)
                cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.predict.links[i]['color'][::-1],
                         thickness=5)

    # -----------------------------  Filters -----------------------------------

    def runVideoFilter(self, debug=False):
        """ WORK IN PROGRESS
        Mystery Function
        """
        thresh = 0.2
        cam = cv2.VideoCapture(self.predict.src)
        self.filter.activated_filters = [0] * self.filter.num_filters
        while True:
            t = time()
            ret_val, img = cam.read()
            img_res, img_hg = self.centerStream(img)
            hg = self.predict.pred(img_hg / 255)
            j = np.ones(shape=(self.predict.params['num_joints'], 2)) * -1
            for i in range(len(j)):
                idx = np.unravel_index(hg[0, :, :, i].argmax(), (64, 64))
                if hg[0, idx[0], idx[1], i] > thresh:
                    j[i] = np.asarray(idx) * 800 / 64
                    if debug:
                        cv2.circle(img_res, center=tuple(j[i].astype(np.int))[::-1], radius=5,
                                   color=self.predict.color[i][::-1], thickness=-1)
            if debug:
                print(j[9])
                self.plotLimbs(img_res, j)
            X = j.reshape((32, 1), order='F')
            _, angles = self.filter.angleAdir(X)
            for f in range(len(self.filter.existing_filters)):
                if np.sum(self.filter.activated_filters) > 0:
                    break
                self.filter.activated_filters[f] = int(eval('self.filter.' + self.filter.existing_filters[f])(angles))
            filter_to_activate = np.argmax(self.filter.activated_filters)
            if self.filter.activated_filters[0] > 0:
                img_res = eval('self.filter.' + self.filter.filter_func[filter_to_activate])(img_res, j)
            fps = 1 / (time() - t)
            cv2.putText(img_res, str(self.filter.activated_filters[0]) + '- FPS: ' + str(fps)[:4], (60, 40), 2, 2,
                        (0, 0, 0), thickness=2)
            cv2.imshow('stream', img_res)
            if cv2.waitKey(1) == 27:
                print('Stream Ended')
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        cam.release()


def predictallimage(params, model='hg_clothes_001_199'):
    """ predict all test image,write into result.csv
    Args:
        params: a dict about config.cfg
        model: the name of the restore model
    """
    inf = InferenceClothes(params, model)
    img_test_dir = params['img_test_dir']
    img_dir_temp = os.path.join(img_test_dir, "Images")
    images = []
    for k in params['category']:
        img_dir_cat = os.path.join(img_dir_temp, k)
        images.extend(os.listdir(img_dir_cat))
    print(images.__len__())
    pre = PredictClothes(params)
    print("Start predicting ...")
    csvresult = open('result_' + strftime('%m%d%H%M') + '.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
    f = open(params['training_txt_file'], 'r')
    firstline = f.readline().strip().split(',')
    f.close()
    writer = csv.writer(csvresult)
    writer.writerow(firstline)
    starttime = time()
    with open(params['test_csv_file'], "r") as f:
        # with open('test_1.csv', "r") as f:
        for value in islice(f, 1, None):  # 读取去掉第一行之后的数据
            value = value.strip().split(',')
            # print(value)
            img_name = value[0]
            cat_temp = value[1]
            print(img_name, cat_temp)
            img = cv2.imread(os.path.join(params['img_test_dir'], img_name))
            height = img.shape[0]
            width = img.shape[1]
            img_resize = cv2.resize(img, (256, 256))
            predjoints = inf.predictJoints(img_resize)
            # predjoints = np.arange(48).reshape((24, 2))
            joints = []
            joints_plot = []  # 传递给plt_skeleton可视化输出结果
            joints.append(img_name)
            joints.append(cat_temp)
            for i in range(predjoints.shape[0]):
                joints.append(
                    str(int(predjoints[i][1] / 256 * width)) + '_' + str(
                        int(predjoints[i][0] / 256 * height)) + '_1')
                joints_plot.append(int(predjoints[i][1] / 256 * width))
                joints_plot.append(int(predjoints[i][0] / 256 * height))
            writer.writerow(joints)
            img_dst = os.path.join(params['log_dir_test'], cat_temp)
            if not os.path.exists(img_dst):
                os.makedirs(img_dst)
            img_plot = pre.plt_skeleton(img, joints_plot, params['joint_list'])
            cv2.imwrite(os.path.join(img_dst, img_name.split('/')[2]), img_plot)
    csvresult.close()
    print("test images in", time() - starttime, " sec")


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 2:
        epoch = argv[1]
    else:
        raise ValueError('need one parameter ,which is the number of epoch\n'
                         'for example: python inferenceclothes.py 100 ')
    params = process_config_clothes()
    print(params)
    starttime = time()
    model = './hourglass_saver/model/' + params['name'] + '/' + params['name'] + "_" + epoch
    print(model)
    predictallimage(params, model)
    # predictallimage(params)
    print("load model and test images in", time() - starttime, " sec")
