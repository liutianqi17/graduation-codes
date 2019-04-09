import pytorch_colors as colors
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


'''
ima = mpimg.imread(r'E:\database\CBSR-Antispoofing\croped\train\1\1_1_1.jpg')
ima_r = ima[:, :, 0]
ima_g = ima[:, :, 1]
ima_b = ima[:, :, 2]
#获取亮度,即原图的灰度拷贝
ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16
#获取蓝色分量
ima_cb = -0.148223 * ima_r - 0.290992 * ima_g + 0.439215 * ima_b + 128
#获取红色分量
ima_cr = 0.439215 * ima_r - 0.367789 * ima_g - 0.071426 * ima_b + 128

# 将三个分量合并在一起
ima_rgb2ycbcr = np.zeros(ima.shape)
ima_rgb2ycbcr[:,:,0] = ima_y
ima_rgb2ycbcr[:,:,1] = ima_cb
ima_rgb2ycbcr[:,:,2] = ima_cr

plt.imshow(ima_rgb2ycbcr)
plt.show()
'''

t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])
tra = transforms.ToPILImage()


pil_img = Image.open(r'E:\database\CBSR-Antispoofing\croped\train\1\1_1_1.jpg')
ten_img = t(pil_img)

hsv = colors.rgb_to_ycbcr(ten_img)
hsv = tra(hsv)
hsv.save('1.jpg')
