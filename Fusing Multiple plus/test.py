import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional
from model import *
import numpy
from sklearn.externals import joblib
from sklearn import svm
import os

rgb = RGB_model()
rgb.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple plus\casia_model\rgb.pkl'))
rgb.eval().cuda()
hsv = HSV_model()
hsv.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple plus\casia_model\hsv.pkl'))
hsv.eval().cuda()
ycbcr = YCbCr_model()
ycbcr.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple plus\casia_model\ycbcr.pkl'))
ycbcr.eval().cuda()
patch = patch_model()
patch.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple plus\casia_model\patch.pkl'))
patch.eval().cuda()
lbp = LBP_model()
lbp.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple plus\casia_model\lbp.pkl'))
lbp.eval().cuda()


path = r'E:/database/nuaa/data/test/0/'
lbp_path = r'E:/database/nuaa/LBP/test/0/'
# path = r'E:/database/CBSR-Antispoofing/croped/test/0/'
# lbp_path = r'E:/database/CBSR-Antispoofing/LBP/test/0/'
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
    lbp_img = Image.open(lbp_path + test_img).convert('RGB')
    img = t(pil_img)
    img.unsqueeze_(dim=0)
    img = Variable(img).cuda()
    imge = tr(pil_img)
    imge.unsqueeze_(dim=0)
    imge = Variable(imge).cuda()
    limg = t(lbp_img)
    limg.unsqueeze_(dim=0)
    limg = Variable(limg).cuda()

    rgb_out = rgb(img)
    rgb_out = functional.softmax(rgb_out, dim=1)
    rgb_out = rgb_out.cpu()
    rgb_out = rgb_out.detach().numpy()
    #rgb_result = int(numpy.argmax(rgb_out.cpu().detach().numpy()))
    hsv_out = hsv(img)
    hsv_out = functional.softmax(hsv_out, dim=1)
    hsv_out = hsv_out.cpu()
    hsv_out = hsv_out.detach().numpy()
    #hsv_result = int(numpy.argmax(hsv_out.cpu().detach().numpy()))
    ycbcr_out = ycbcr(img)
    ycbcr_out = functional.softmax(ycbcr_out, dim=1)
    ycbcr_out = ycbcr_out.cpu()
    ycbcr_out = ycbcr_out.detach().numpy()
    #ycbcr_result = int(numpy.argmax(ycbcr_out.cpu().detach().numpy()))
    patch_out = patch(imge)
    patch_out = functional.softmax(patch_out, dim=1)
    patch_out = patch_out.cpu()
    patch_out = patch_out.detach().numpy()
    #patch_result = int(numpy.argmax(patch_out.cpu().detach().numpy()))
    lbp_out = lbp(limg)
    lbp_out = functional.softmax(lbp_out, dim=1)
    lbp_out = lbp_out.cpu()
    lbp_out = lbp_out.detach().numpy()

    #out = out.detach().numpy()
    #ind = int(numpy.argmax(out))
    # ind = out[0, 0]+out[0, 1]
    #print(rgb_out, hsv_out, ycbcr_out, patch_out)
    #print(rgb_result, hsv_result, ycbcr_result, patch_result)


    clf = joblib.load(r"C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple plus\casia_model\svm.m")
    test_X = [[rgb_out[0, 0], hsv_out[0, 0], ycbcr_out[0, 0], lbp_out[0, 0], patch_out[0, 0]]]
    #print(clf.predict(test_X)[0])
    if clf.predict(test_X)[0] == 0:
        num_true +=1
    print(num_true)

