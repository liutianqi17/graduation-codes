import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional
from model import *
import numpy
from data import *

[test_loader, test_dataset] = loadtestdata(patch=True, batch=1)

rgb = RGB_model()
rgb.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\pkl\rgb_net.pkl'))
rgb.eval().cuda()
hsv = HSV_model()
hsv.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\pkl\hsv_net.pkl'))
hsv.eval().cuda()
ycbcr = YCbCr_model()
ycbcr.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\pkl\ycbcr_net.pkl'))
ycbcr.eval().cuda()
patch = patch_model()
patch.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\pkl\patch.pkl'))
patch.eval().cuda()

numpy.set_printoptions(suppress=True)


'''
prob_file = open('prob_file.txt', 'w')
for data in test_loader:
    img, label = data
    img = Variable(img).cuda()
    rgb_out = rgb(img)
    rgb_out = functional.softmax(rgb_out)
    rgb_result = rgb_out.cpu().detach().numpy()
    hsv_out = hsv(img)
    hsv_out = functional.softmax(hsv_out)
    hsv_result = hsv_out.cpu().detach().numpy()
    ycbcr_out = ycbcr(img)
    ycbcr_out = functional.softmax(ycbcr_out)
    ycbcr_result = ycbcr_out.cpu().detach().numpy()
    # patch_out = rgb(patch_img)
    # patch_out = functional.softmax(patch_out)
    # patch_result = patch_out.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    prob_file.writelines([str(rgb_result[0, 0]), ',',
                          str(hsv_result[0, 0]), ',',
                          str(ycbcr_result[0, 0]), ',',
                          str(label[0]), '\n'])
prob_file.close()
'''
prob_file = open('patch.txt', 'w')
for data in test_loader:
    img, label = data
    img = Variable(img).cuda()
    patch_out = rgb(img)
    patch_out = functional.softmax(patch_out)
    patch_result = patch_out.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    prob_file.writelines([str(patch_result[0, 0]), ',', str(label[0]), '\n'])
prob_file.close()