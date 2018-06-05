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

from keras.utils import plot_model

model = load_model('model/weights.05-1.00.hdf5', custom_objects={"Scale": Scale})
plot_model(model, to_file='model.png')

