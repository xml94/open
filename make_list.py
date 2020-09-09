'''
Read desired label's image ID
From COCO dataset
'''

from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import shutil
import cv2


def check_dir(abs_dir):
    if os.path.exists(abs_dir):
        print('%s already exists' % abs_dir)
    else:
        os.mkdir(abs_dir)


root = os.getcwd()


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"

# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
# -------------- Please change these parameters ---------------------- #

dataDir = '/home/multiai3/Mingle/GAN/tomato_leaves_translation'
dir_change = ['data_dataset_canker']

classes = ['canker']

percentage = 0.1

# -------------- Please change nothing following here ---------------------- #


for dataType in dir_change: #, 'dataset_coco_health'
    annFile = '{}/{}/annotations.json'.format(dataDir, dataType)
    coco = COCO(annFile)

    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    for className in classes:
        catIds = coco.getCatIds(catNms=className)
        imgIds = coco.getImgIds(catIds=catIds)

        save_dir = os.path.join('dataset', className)
        img_dir = os.path.join(save_dir, 'image_val' if 'val' in dataType else 'image_train')
        mask_dir = os.path.join(save_dir, 'mask_val' if 'val' in dataType else 'mask_train')
        check_dir(save_dir)
        check_dir(img_dir)
        check_dir(mask_dir)

        for id in imgIds:
            info = coco.loadImgs(id)[0]
            img_source_name = '{}/{}/{}'.format(dataDir, dataType, info['file_name'])
            # print(img_dir)
            # print(info['file_name'].replace('JPEGImages/', ''))
            img_target_name = os.path.join(dataDir, img_dir, info['file_name'].replace('JPEGImages/', ''))

            # img = cv2.imread(img_source_name)

            # save mask. when multiple mask, use accumulation number
            annIds = coco.getAnnIds(imgIds=info['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            valid_number = 0
            flag = True
            for i in range(len(anns)):
                classname = getClassName(anns[i]['category_id'], cats)

                if classname == className:
                    # copy image only once
                    if flag:
                        # print(img_source_name)
                        # print(img_target_name)
                        shutil.copyfile(img_source_name, img_target_name)
                        flag = False

                    # make mask
                    mask = coco.annToMask(anns[i])
                    name, _ = os.path.splitext(info['file_name'])

                    H, W = mask.shape
                    if np.sum(mask) > percentage * H * W:
                        mask_name = os.path.join(mask_dir, name.replace('JPEGImages/', '') + '_mask_' + str(valid_number) + '.jpg')
                        plt.imsave(mask_name, mask, cmap='gray')
                        valid_number += 1

                    # mask_name = os.path.join(mask_dir, name + '_mask_' + str(valid_number) + '.jpg')
                    # plt.imsave(mask_name, mask, cmap='gray')
                    # valid_number += 1

        print('Successfully for %s' % className)


