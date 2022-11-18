import os
from skimage import measure
from matplotlib.pyplot import imread
import numpy as np

ceba_path = './celeba/Img/img_align_celeba/'
fake_path = './haoran_1/'

from skimage.metrics import structural_similarity as compare_ssim

def cal_ssim(img1_path, img2_path):
    img1 = imread(img1_path)
    img2 = imread(img2_path)
    img1 = np.resize(img1, (img2.shape[0], img2.shape[1], img2.shape[2]))
    ssim = compare_ssim(img1, img2, channel_axis=-1)
    return ssim

def read_all_file_name(file_path):
    file_name = os.listdir(file_path)
    return file_name

fake_name = read_all_file_name(fake_path)
true_name = read_all_file_name(ceba_path)
# print(true_name)

for fake_file_name in fake_name:
    max_ssim = -1
    max_fake_file = fake_path + fake_file_name
    max_true_file = ''
    print(fake_file_name)
    for true_file_name in true_name:
#         print(true_file_name)
#         print(max_ssim)
        ssim = cal_ssim(fake_path + fake_file_name, ceba_path + true_file_name)
#         print(ssim)
        if ssim > max_ssim:
            max_ssim = ssim
            max_true_file = ceba_path + true_file_name
#         print('max_true_file is ' + max_true_file)
    print(max_fake_file + '====' + max_true_file)
    max_true_file = ''
