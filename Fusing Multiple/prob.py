import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional
from model import *
import numpy
from data import *

[test_loader, test_dataset] = loadtestdata(patch=True, batch=1)

'''
rgb = RGB_model()
rgb.load_state_dict(torch.load(r'C:/Users/Neticle/Desktop/bishedaima/Fusing Multiple/casia_model/rgb.pkl'))
rgb.eval().cuda()
hsv = HSV_model()
hsv.load_state_dict(torch.load(r'C:/Users/Neticle/Desktop/bishedaima/Fusing Multiple/casia_model/hsv.pkl'))
hsv.eval().cuda()
ycbcr = YCbCr_model()
ycbcr.load_state_dict(torch.load(r'C:/Users/Neticle/Desktop/bishedaima/Fusing Multiple/casia_model/ycbcr.pkl'))
ycbcr.eval().cuda()
'''
patch = []
for i in range(10):
    patch.append(patch_model())
    patch[i].load_state_dict(torch.load('C:/Users/Neticle/Desktop/bishedaima/Fusing Multiple/nuaa_model/patch'
                                     + str(i) + '.pkl'))
    patch[i].eval().cuda()

numpy.set_printoptions(suppress=True)


prob_file = open('patch.txt', 'w')
for data in test_loader:
    img, label = data
    img = Variable(img).cuda()
    patch_out = []
    patch_result = []
    for j in range(10):
        patch_out.append(patch[j](img))
        patch_out[j] = functional.softmax(patch_out[j], dim=1)
        patch_result.append(patch_out[j].cpu().detach().numpy())
        prob_file.writelines([str(patch_result[j][0, 0]), ','])
    label = label.cpu().detach().numpy()
    prob_file.writelines([str(label[0]), '\n'])
prob_file.close()
