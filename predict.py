# -*- coding: utf-8 -*-

import os
import sys

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from custom_layers.scale_layer import Scale

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

TEST_DIR = '/home/asilla/thang/tunnel_in/valid'
LABELS = {'other': 0, 'tunnel_in': 1}

if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    model = load_model('model/weights.24-0.88.hdf5', custom_objects={"Scale": Scale})
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    results = {'other':{}, 'tunnel_in': {}}
    images_folder = os.listdir(TEST_DIR)
    for folder in images_folder:
      images_dir = os.path.join(TEST_DIR, folder)
      imagefiles = os.listdir(images_dir)
      results[folder]['total'] = len(imagefiles)
      results[folder]['correct'] = 0
      for imagefile in imagefiles:
        image_path = os.path.join(images_dir, imagefile)
        img = image.load_img(image_path, target_size=(img_cols, img_rows))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        y_proba = model.predict(images, batch_size=1)
        y_classes = y_proba.argmax(axis=-1)
        if y_classes[0] == LABELS[folder]:
          results[folder]['correct'] = results[folder]['correct'] + 1
    print results
