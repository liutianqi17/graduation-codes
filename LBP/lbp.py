import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import PIL
import os


def circular_LBP(src, radius, n_points):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)

    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.
            # 先计算共n_points个点对应的像素值，使用双线性插值法
            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)

                # 向下取整
                x1 = int(math.floor(x_n))
                y1 = int(math.floor(y_n))
                # 向上取整
                x2 = int(math.ceil(x_n))
                y2 = int(math.ceil(y_n))

                # 将坐标映射到0-1之间
                tx = np.abs(x - x1)
                ty = np.abs(y - y1)

                # 根据0-1之间的x，y的权重计算公式计算权重
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbour = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                neighbours[0, n] = neighbour

            center = src[y, x]

            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0

            for n in range(n_points):
                lbp += lbp_value[0, n] * 2**n

            # 转换到0-255的灰度空间，比如n_points=16位时结果会超出这个范围，对该结果归一化
            dst[y, x] = int(lbp / (2**n_points-1) * 255)

    return dst


path = r'E:/database/nuaa/data/train/0/'
target_path = r'E:/database/nuaa/LBP/train/0/'
num = 1
for img_path in os.listdir(path):
    if num <= 4466:
        num += 1
        continue
    img = Image.open(path + img_path).convert('L')
    # img = img.resize((256, 256), Image.ANTIALIAS)
    result = Image.fromarray(circular_LBP(np.array(img), radius=1, n_points=8))
    result.save(target_path + img_path)

'''
path = r'E:/database/CBSR-Antispoofing/croped/train/0/'
target_path = r'E:/database/CBSR-Antispoofing/LBP/train/0/'
for img_path in os.listdir(path):
    img = Image.open(path + img_path).convert('L')
    result = Image.fromarray(circular_LBP(np.array(img), radius=1, n_points=8))
    result.save(target_path + img_path)


root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename()

img = Image.open(path).convert('L')

img = Image.fromarray(circular_LBP(np.array(img), radius=1, n_points=8))
img.show()
'''
