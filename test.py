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

TEST_DIR = '/home/asilla/acr/data/duplicate/'
# RESULTS_DIR = '/home/khach/hanh/acr/keras_sources/images/reuslt/'
LABELS = {0:'0_backview', 1:'1_fontview', 2:'2_sideview'}

if __name__ == '__main__':

  img_rows, img_cols = 224, 224
  model = load_model('model/weights.50-0.96217105.hdf5', custom_objects={"Scale": Scale})
  nadam = Nadam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
  model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
  count = 0
  #count_error = 0
  count_1 = 0
  count_2 = 0
  count_3 = 0
  Dir = ['carview','goonet','kakaku']
  for webname in Dir:
    path_ = os.path.join(TEST_DIR, webname)
    images_folder = os.listdir(path_)
    for subfolder_1 in images_folder:
        for subfolder_2 in os.listdir(os.path.join(path_,subfolder_1)):
          # print(subfolder_2)
          for folder in os.listdir(os.path.join(path_, subfolder_1, subfolder_2 )):
            # print(folder)
            #exit()
            images_dir = os.path.join(path_, subfolder_1, subfolder_2 , folder)
            #print(images_dir)
            #exit()
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

              y_proba = model.predict(np.array(images))
              y_classes = y_proba.argmax(axis=-1)

              for f, y in zip(fnames, y_classes):
                count +=1
                src = os.path.join(path_, subfolder_1, subfolder_2, folder, f)

                if LABELS[y] == '0_backview':
                  backview_dest = src.replace('/{}'.format(webname), '/filter/filter_back')
                  if not os.path.exists(os.path.dirname(backview_dest)):
                    os.makedirs(os.path.dirname(backview_dest))

                  count_1 +=1
                  print("backview",src)
                  #print("backview_new", os.path.join(os.path.dirname(backview_dest) , f))
                  shutil.copy(src, os.path.dirname(backview_dest))

                elif LABELS[y] == '1_fontview':
                  fontview_dest = src.replace('/{}'.format(webname), '/filter/filter_font')
                  if not os.path.exists(os.path.dirname(fontview_dest)):
                    os.makedirs(os.path.dirname(fontview_dest))
                  print("fontview", src)
                  count_2 += 1
                  #print("fontview_new", os.path.join(os.path.dirname(fontview_dest) , f))
                  shutil.copy(src, os.path.dirname(fontview_dest))
                else:
                  sideview_dest = src.replace('/{}'.format(webname), '/filter/filter_side')
                  if not os.path.exists(os.path.dirname(sideview_dest)):
                    os.makedirs(os.path.dirname(sideview_dest))

                  count_3 +=1
                  print("sideview", src)
                  #print("sideview_new", os.path.join(os.path.dirname(sideview_dest) , f))
                  shutil.copy(src, os.path.dirname(sideview_dest))

    print("backview : {}".format(count_1) , "fontview : {}".format(count_2) , "sideview : {}".format(count_3))
    #print("result {}%".format(float(count_2)*100/(float(count_2) + float(count_1))))
    print("tong so anh : {}".format(count))
