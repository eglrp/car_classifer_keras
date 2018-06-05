from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid

def generate_data_txt():
    data_dir = '/home/thangdt/Documents/MC/acr/car_view/images'
    split_folders = next(os.walk(data_dir))[1]
    for split in split_folders:
        tunnel_dir = os.path.join(data_dir, split)
        tunnel_folders = next(os.walk(tunnel_dir))[1]
        for tunnel in tunnel_folders:
            image_dir = os.path.join(tunnel_dir, tunnel)
            imagefiles = os.listdir(image_dir)
            for imagefile in imagefiles:
                src = os.path.join(image_dir, imagefile)
                dst = os.path.join(image_dir, uuid.uuid4().hex + '.jpg')
                os.rename(src, dst)

if __name__ == '__main__':
    generate_data_txt()