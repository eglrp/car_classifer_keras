# -*- coding: utf-8 -*-

import os
import sys
import shutil

from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import glob
from custom_layers.scale_layer import Scale

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

TEST_DIR = '/home/buiduchanh/WorkSpace/keras_sources/acr-view_KERAS_BAK/images'
RESULTS_DIR = '/home/khach/hanh/acr/keras_sources/images/reuslt/'
ERR_DIR = '/home/khach/hanh/acr/keras_sources/images/error/'
LABELS = {0:'0_backview', 1:'1_fontview', 2:'2_sideview'}

if __name__ == '__main__':
  img_rows, img_cols = 224, 224
  model = load_model('model/weights.21-0.88502994.hdf5', custom_objects={"Scale": Scale})
  nadam = Nadam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
  model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])

  images_folder = os.listdir(TEST_DIR)
  count_1 = 0
  count_2 = 0
  for folder in images_folder:
    images_dir = os.path.join(TEST_DIR, folder)
    imagefiles = glob.glob('%s/*' % images_dir)
    batch_size = 500
    for block in range(0, len(imagefiles), batch_size):
      images = []
      fnames = []
      for pos in range(0, min(batch_size, len(imagefiles) - block)):
        filename = ''
        try:
          filename = os.path.basename(imagefiles[pos + block])
          img = image.load_img(imagefiles[pos + block], target_size=(img_cols, img_rows))
          x = image.img_to_array(img)
          images.append(x)
          fnames.append(filename)
        except:
          print("error")
          shutil.copy(imagefiles[pos + block], os.path.join(ERR_DIR, filename))

      y_proba = model.predict(np.array(images))
      print(y_proba)
      exit()
      y_classes = y_proba.argmax(axis=-1)
      for f, y in zip(fnames, y_classes):
        src = os.path.join(TEST_DIR, folder, f)
        err_dest = os.path.join(ERR_DIR, folder, LABELS[y], f)
        if not os.path.exists(err_dest):
          os.makedirs(err_dest)
        result_dest = os.path.join(RESULTS_DIR, folder, f)
        if not os.path.exists(result_dest):
          os.makedirs(result_dest)

        if LABELS[y] != folder:
          count_1 +=1
          print("right",src)
          shutil.copy(src, err_dest)
        else:
          count_2 +=1
          shutil.copy(src, result_dest)
  print("result {}%".format(float(count_2)*100/(float(count_2) + float(count_1))))