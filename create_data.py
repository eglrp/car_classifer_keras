from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from shutil import copyfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = '/home/asilla/hanh/acr/keras_sources/images'
FILTER_DIR = '/home/asilla/hanh/acr/keras_sources/data'

def generate_image():
    split_folders = next(os.walk(DATA_DIR))[1]
    for split in split_folders:
        side_filter = os.path.join(FILTER_DIR, split + '_filter')
        if not os.path.exists(side_filter):
            os.makedirs(side_filter)
        side_dir = os.path.join(DATA_DIR, split)
        side_views = sorted(next(os.walk(side_dir))[1])
        count = 0
        for side_view in side_views:
            image_dir = os.path.join(side_dir, side_view)
            imagefiles = os.listdir(image_dir)
            for imagefile in imagefiles:
                image_path = os.path.join(image_dir, imagefile)
                crop_image_path = os.path.join(side_filter, str(count) + '_' + imagefile)
                copyfile(image_path, crop_image_path)
            count = count + 1

if __name__ == '__main__':
    generate_image()
