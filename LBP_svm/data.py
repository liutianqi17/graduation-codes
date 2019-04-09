import os
from PIL import Image
import numpy as np
import math

prob_file = open('train_data.txt', 'w')

num = 0
path = r'E:/database/CBSR-Antispoofing/LBP/train/0/'
for test_img in os.listdir(path):
    print(num)
    num += 1
    pil_img = Image.open(path + test_img).convert('L')
    pil_img = pil_img.resize((64, 64), Image.ANTIALIAS)
    img = np.array(pil_img)
    histograms = [0 for i in range(512)]
    for i in range(4):
        for j in range(4):
            for x_pix in range(i*16, i*16+16):
                for y_pix in range(j * 16, j * 16 + 16):
                    level = math.floor(img[x_pix, y_pix]/8)
                    histograms[(i*4+j)*32+level] += 1
    for k in range(512):
        prob_file.writelines([str(histograms[k]), ','])
    prob_file.writelines([str(0), '\n'])


path = r'E:/database/CBSR-Antispoofing/LBP/train/1/'
for test_img in os.listdir(path):
    print(num)
    num += 1
    pil_img = Image.open(path + test_img).convert('L')
    pil_img = pil_img.resize((64, 64), Image.ANTIALIAS)
    img = np.array(pil_img)
    histograms = [0 for i in range(512)]
    for i in range(4):
        for j in range(4):
            for x_pix in range(i*16, i*16+16):
                for y_pix in range(j * 16, j * 16 + 16):
                    level = math.floor(img[x_pix, y_pix]/8)
                    histograms[(i*4+j)*32+level] += 1
    for k in range(512):
        prob_file.writelines([str(histograms[k]), ','])
    prob_file.writelines([str(1), '\n'])

prob_file.close()