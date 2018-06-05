from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import uuid
import pyyolo
from shutil import copyfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DARKNET_PATH = '/home/phoenix/thang/pyyolo/darknet'
DATA_CFG = 'cfg/voc.data'
CFG_FILE = 'cfg/yolo-voc.cfg'
WEIGHT_FILE = 'yolo-voc.weights'

DATA_DIR = '/home/phoenix/thang/acr-view/valid'

def crop_image(image_path):
    thresh = 0.24
    hier_thresh = 0.5
    areas = []
    coords = []

    im = cv2.imread(image_path)
    height, width, _ = im.shape
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
        os.remove(image_path)
        return False

    car_coord = coords[areas.index(max(areas))]
    scale_width = int((car_coord[2] - car_coord[0]) * 0.05)
    scale_height = int((car_coord[3] - car_coord[1]) * 0.05)
    car_coord[0] = car_coord[0] - scale_width
    car_coord[1] = car_coord[1] - scale_height
    car_coord[2] = car_coord[2] + scale_width
    car_coord[3] = car_coord[3] + scale_height
    car_coord = [max(int(car_coord[0]), 2), max(int(car_coord[1]), 2), car_coord[2], car_coord[3]]
    new_image = im[car_coord[1]:car_coord[3], car_coord[0]:car_coord[2]]
    cv2.imwrite(image_path, new_image)

def generate_image():
    split_folders = next(os.walk(DATA_DIR))[1]
    for split in split_folders:
        image_dir = os.path.join(DATA_DIR, split)
        imagefiles = os.listdir(image_dir)
        for imagefile in imagefiles:
            image_path = os.path.join(image_dir, imagefile)
            crop_image(image_path)

if __name__ == '__main__':
    pyyolo.init(DARKNET_PATH, DATA_CFG, CFG_FILE, WEIGHT_FILE)
    generate_image()
