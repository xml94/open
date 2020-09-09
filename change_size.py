import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def check_dir(abs_dir):
    if os.path.exists(abs_dir):
        pass
    else:
        os.mkdir(abs_dir)
        print('Make new director: %s' % abs_dir)


src_dir = '/home/multiai3/Mingle/GAN/tomato_leaves_translation/dataset'
target_dir = '/home/multiai3/Mingle/GAN/tomato_leaves_translation/dataset/0904_health_canker'
check_dir(target_dir)
size = (256, 256)

classes = ['canker']


for className in classes:
    source_dir = os.path.join(src_dir, className)
    for i in sorted(os.listdir(source_dir)):
        init_abs_dir = os.path.join(src_dir, className, i)
        target_abs_dir = os.path.join(target_dir, className, i)
        check_dir(os.path.join(target_dir, className))
        check_dir(target_abs_dir)

        file_list = os.listdir(init_abs_dir)
        number_valid = 0
        for file_name in file_list:
            init_abs_name = os.path.join(init_abs_dir, file_name)
            target_abs_name = os.path.join(target_abs_dir, file_name)

            if 'image' in i:
                init_img = cv2.imread(init_abs_name)
                target_img = cv2.resize(init_img, size)
                cv2.imwrite(target_abs_name, target_img)
            elif 'mask' in i:
                init_mask = cv2.imread(init_abs_name, 0)
                target_mask = cv2.resize(init_mask, size)
                plt.imsave(target_abs_name, target_mask, cmap='gray')

        print('Successfully resize for %s %s' % (className, i))
