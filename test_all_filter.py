# -*- coding: utf-8 -*-

import os
import sys
import shutil
import pyyolo
import cv2
import traceback
import numpy as np

from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from custom_layers.scale_layer import Scale
from keras.models import load_model
from keras.preprocessing import image

IMG_ROWS = 224
IMG_COLS = 224
LABELS = {0:'front-view', 1:'back-view', 2:'side-view', 3:'other'}
IMAGES_FILTER = '/home/phoenix/thang/acr-view/test_1'
RESULTS_DIR = '/home/phoenix/thang/acr-view/filter_view_test'
DARKNET_PATH = '/home/phoenix/thang/pyyolo/darknet'
DATA_CFG = 'cfg/voc.data'
CFG_FILE = 'cfg/yolo-voc.cfg'
WEIGHT_FILE = 'yolo-voc.weights'

def load_image(image_path):
    try:
        thresh = 0.24
        hier_thresh = 0.5
        areas = []
        coords = []

        img = cv2.imread(image_path)
        height, width, _ = img.shape
        area = width * height
        outputs = pyyolo.test(image_path, thresh, hier_thresh, 0)

        for out in outputs:
            if out['class'] == 'car':
                if out['prob'] < 0.65:
                    continue

                x1 = out['left']
                x2 = out['right']
                y1 = out['top']
                y2 = out['bottom']

                if abs((x2 - x1) * (y2 - y1)) < 0.3 * area:
                    continue

                coords.append([x1, y1, x2, y2])
                areas.append(abs((x2 - x1) * (y2 - y1)))

        if not coords:
            return False
        car_coord = coords[areas.index(max(areas))]
        
        scale_width = int((car_coord[2] - car_coord[0]) * 0.1)
        scale_height = int((car_coord[3] - car_coord[1]) * 0.1)
        car_coord[0] = car_coord[0] - scale_width
        car_coord[1] = car_coord[1] - scale_height
        car_coord[2] = car_coord[2] + scale_width
        car_coord[3] = car_coord[3] + scale_height
        car_coord = (max(int(car_coord[0]), 2), max(int(car_coord[1]), 2), min(int(car_coord[2]), width - 2), min(int(car_coord[3]), height - 2))

        crop_img = img[car_coord[1]:car_coord[3], car_coord[0]:car_coord[2]]
        return crop_img
    except:
        print("Load Error: %s" % image_path)
        traceback.print_exc()
        return None


def filter_image():
    weight = load_model('model/weights.28-0.98500000.hdf5', custom_objects={"Scale": Scale})
    nadam = Nadam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    weight.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])

    makers_img = os.walk(IMAGES_FILTER).next()[1]
    for maker in sorted(makers_img):
        models_dir = os.path.join(IMAGES_FILTER, maker)
        models_img = os.walk(models_dir).next()[1]
        for model in sorted(models_img):
            years_dir = os.path.join(models_dir, model)
            years_img = os.walk(years_dir).next()[1]

            index = {0:1, 1:1, 2:1, 3:1}
            for year in sorted(years_img):
                images_dir = os.path.join(years_dir, year)
                for imagefile in sorted(os.listdir(images_dir)):
                    try:
                        image_path = os.path.join(images_dir, imagefile)
                        img = load_image(image_path)
                        if img is None:
                            continue

                        cv2.imwrite(RESULTS_DIR + "/result.jpg", img)

                        resize_img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
                        x = np.expand_dims(resize_img, axis=0)

                        images = np.vstack([x])
                        y_proba = weight.predict(images, batch_size=1)
                        y_classes = y_proba.argmax(axis=-1)
                        y_percent = y_proba[0][y_classes[0]]
                        print("Class: %s %s %s", (imagefile, y_classes, y_percent))
                    except:
                        print("Filter Error: %s" % os.path.join(images_dir, imagefile))
                        traceback.print_exc()
if __name__ == '__main__':
    pyyolo.init(DARKNET_PATH, DATA_CFG, CFG_FILE, WEIGHT_FILE)
    filter_image()

