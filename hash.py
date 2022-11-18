import os
from skimage import measure
from matplotlib.pyplot import imread
import numpy as np
import cv2

ceba_path = './celeba/Img/img_align_celeba/'
fake_path = './haoran_1/'

def read_all_file_name(file_path):
    file_name = os.listdir(file_path)
    return file_name


fake_name = read_all_file_name(fake_path)
true_name = read_all_file_name(ceba_path)

def cmpHash(hash1, hash2,shape=(10,10)):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 相等则n计数+1，n最终为相似度
        if hash1[i] == hash2[i]:
            n = n + 1
    return n/(shape[0]*shape[1])

def aHash(img,shape=(10,10)):
    # 缩放为10*10
    img = cv2.resize(img, shape)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(shape[0]):
        for j in range(shape[1]):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 100
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

def cal_hash(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    hash1 = aHash(img1)
    hash2 = aHash(img2)

    n = cmpHash(hash1, hash2)

    return n


for fake_file_name in fake_name:
    max_ssim = -1
    max_fake_file = fake_path + fake_file_name
    max_true_file = ''
    for true_file_name in true_name:
#         print(true_file_name)
#         print(max_ssim)
        ssim = cal_hash(fake_path + fake_file_name, ceba_path + true_file_name)
#         print(ssim)
        if ssim > max_ssim:
            max_ssim = ssim
            max_true_file = ceba_path + true_file_name
#         print('max_true_file is ' + max_true_file)
    print(max_fake_file + '====' + max_true_file)
    max_true_file = ''
