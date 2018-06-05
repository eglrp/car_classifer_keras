
import sys
import os
import cv2
import numpy as np
import glob

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

num_classes = 3

def load_data(img_rows, img_cols):

    # Load train data
    X_train = []
    Y_train = []
    for img_path in glob.glob('data/train_filter/*.jpg'):
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_rows,img_cols))
        X_train.append(img)
        Y_train.append([int(os.path.basename(img_path).split('_')[0])])

    X_valid = []
    Y_valid = []
    for img_path in glob.glob('data/valid_filter/*.jpg'):
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_rows,img_cols))
        X_valid.append(img)
        Y_valid.append([int(os.path.basename(img_path).split('_')[0])])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_valid.shape)
    print(Y_valid.shape)

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)

    return X_train, Y_train, X_valid, Y_valid
