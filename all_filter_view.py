# -*- coding: utf-8 -*-

import os
import sys
import shutil
import pyyolo
import numpy as np

from PIL import ImageFile
from PIL import Image as pil_image
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

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_ROWS = 224
IMG_COLS = 224
LABELS = {0:'front-view', 1:'back-view', 2:'side-view', 3:'other'}
IMAGES_FILTER = '/home/phoenix/thang/acr-crawler/images'
RESULTS_DIR = '/home/phoenix/thang/acr-view/filter_view'
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

        img = pil_image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size
        area = width * height
        outputs = pyyolo.test(image_path, thresh, hier_thresh, 0)
        print outputs

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

        crop_img = img.crop(car_coord)
        return crop_img
    except:
        print("Load Error: %s" % image_path)
        return False

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
                        if img == False:
                            continue

                        resize_img = img.resize((IMG_COLS, IMG_ROWS))
                        x = image.img_to_array(resize_img)
                        x = np.expand_dims(x, axis=0)

                        images = np.vstack([x])
                        y_proba = weight.predict(images, batch_size=1)
                        y_classes = y_proba.argmax(axis=-1)
                        y_percent = y_proba[0][y_classes[0]]
                        if y_percent < 0.65:
                            continue

                        folder_dest = os.path.join(RESULTS_DIR, LABELS[y_classes[0]], maker + '|' + model)
                        if not os.path.exists(folder_dest):
                            os.makedirs(folder_dest)
                        
                        new_file = '0'*(6 - len(str(index[y_classes[0]]))) + str(index[y_classes[0]]) + ".jpg"
                        new_img_dest = os.path.join(folder_dest, new_file)

                        img.save(new_img_dest)
                        index[y_classes[0]] = index[y_classes[0]] + 1
                    except:
                        print("Filter Error: %s" % os.path.join(images_dir, imagefile))
if __name__ == '__main__':
    pyyolo.init(DARKNET_PATH, DATA_CFG, CFG_FILE, WEIGHT_FILE)
    #filter_image()
    load_image('DBA-NZE144G_0234941.jpg')

