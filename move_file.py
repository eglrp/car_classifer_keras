import glob
import os
import shutil
import random

TRANDIR = '/media/buiduchanh/Work/Workspace/acr/data/train'
# TESTDIR = '/media/buiduchanh/Work/Workspace/acr/data_classification/test'
# VALIDDIR = '/media/buiduchanh/Work/Workspace/acr/data_classification/valid'

for model in os.listdir(TRANDIR):
    numimages = sorted(glob.glob('{}/**'.format(os.path.join(TRANDIR, model))))
    numTrain = random.sample(numimages, round(len(numimages)/5) -1 )
    for images in numTrain:
        newpath = os.path.dirname(images).replace('/train', '/car_test')
        print(newpath)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        shutil.move(images,newpath)