from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# MODELS = {'Honda': ['Fit','Fit_Hybrid'], 'Mazda': ['Axela', 'Axela_Sport'], 'Subaru': ['Impreza', 'Impreza_G4'], 'Subaru': ['Impreza_Hatchback', 'Impreza_Hatchback_STI'], 'Toyota': ['Crown', 'Crown_Hybrid'], 'Toyota': ['Harrier', 'Harrier_Hybrid']}
MODELS = {'Toyota': ['Prius', 'Prius_Alpha']}

def merge_files():
    data_dir = '/home/phoenix/thang/acr-crawler/images'
    for key, value in MODELS.iteritems():
        year_path = os.path.join(data_dir, key, value[1])
        years = os.listdir(year_path)
        for year in years:
            if not os.path.exists(os.path.join(data_dir, key, value[0], year)):
                os.makedirs(os.path.join(data_dir, key, value[0], year))
            image_path = os.path.join(year_path, year)
            images = os.listdir(image_path)
            for image in images:
                dest = os.path.join(data_dir, key, value[0], year, '0_' + image)
                src_path = os.path.join(year_path, year, image)
                shutil.move(src_path, dest)
        shutil.rmtree(year_path)

if __name__ == '__main__':
    merge_files()
