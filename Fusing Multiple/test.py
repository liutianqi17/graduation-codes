import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional
from model import *
import numpy
from data import *
from sklearn.externals import joblib
from sklearn import svm
import os

rgb = RGB_model()
rgb.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\nuaa_model\rgb.pkl'))
rgb.eval().cuda()
hsv = HSV_model()
hsv.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\nuaa_model\hsv.pkl'))
hsv.eval().cuda()
ycbcr = YCbCr_model()
ycbcr.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\nuaa_model\ycbcr.pkl'))
ycbcr.eval().cuda()
patch = []
for i in range(10):
    patch.append(patch_model())
    patch[i].load_state_dict(torch.load(
        r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\nuaa_model\patch' + str(i) + '.pkl'))
    patch[i].eval().cuda()


# path = r'E:/database/nuaa/data/test/0/'
# path = r'E:/database/CBSR-Antispoofing/fusing/test2/1/'
path = r'E:/database/CBSR-Antispoofing/croped/test/0/'
t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])
tr = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(96),
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

num_true = 0
for test_img in os.listdir(path):
    pil_img = Image.open(path + test_img).convert('RGB')
    img = t(pil_img)
    img.unsqueeze_(dim=0)
    img = Variable(img).cuda()

    rgb_out = rgb(img)
    rgb_out = functional.softmax(rgb_out, dim=1)
    rgb_out = rgb_out.cpu()
    rgb_out = rgb_out.detach().numpy()
    hsv_out = hsv(img)
    hsv_out = functional.softmax(hsv_out, dim=1)
    hsv_out = hsv_out.cpu()
    hsv_out = hsv_out.detach().numpy()
    ycbcr_out = ycbcr(img)
    ycbcr_out = functional.softmax(ycbcr_out, dim=1)
    ycbcr_out = ycbcr_out.cpu()
    ycbcr_out = ycbcr_out.detach().numpy()

    patch_out = []
    for j in range(10):
        imge = tr(pil_img)
        imge.unsqueeze_(dim=0)
        imge = Variable(imge).cuda()
        patch_out.append(patch[j](imge))
        patch_out[j] = functional.softmax(patch_out[j], dim=1)
        patch_out[j] = patch_out[j].cpu()
        patch_out[j] = patch_out[j].detach().numpy()


    clf = joblib.load(r"C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\nuaa_model\svm.m")
    test_X = [rgb_out[0, 0], hsv_out[0, 0], ycbcr_out[0, 0]]
    for j in range(10):
        test_X.append(patch_out[j][0, 0])
    test_X = [test_X]
    if clf.predict(test_X)[0] == 0:
        num_true += 1
    #else:
        #pil_img.save("img/" + test_img)
    print(num_true)

