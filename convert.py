from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from shutil import copyfile

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_data_txt():
    data_dir = '/home/phoenix/thang/car_view/images'
    filter_dir = '/home/phoenix/thang/car_view/data'
    split_folders = next(os.walk(data_dir))[1]
    for split in split_folders:
        tunnel_dir = os.path.join(data_dir, split)
        tunnel_filter = os.path.join(filter_dir, split + '_filter')
        if not os.path.exists(tunnel_filter):
            os.makedirs(tunnel_filter)

        tunnel_folders = next(os.walk(tunnel_dir))[1]
        count = 0
        for tunnel in tunnel_folders:
            image_dir = os.path.join(tunnel_dir, tunnel)
            imagefiles = os.listdir(image_dir)
            for imagefile in imagefiles:
                try:
                    src = os.path.join(image_dir, imagefile)
                    im = Image.open(src).convert('RGB')
                    dst = os.path.join(tunnel_filter, str(count) + '_' + imagefile)
                    im.save(dst, "JPEG")
                except:
                    print(src)
            count = count + 1

if __name__ == '__main__':
    generate_data_txt()
