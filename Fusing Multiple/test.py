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


rgb = RGB_model()
rgb.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\trained_model\rgb_net.pkl'))
rgb.eval().cuda()
hsv = HSV_model()
hsv.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\trained_model\hsv_net.pkl'))
hsv.eval().cuda()
ycbcr = YCbCr_model()
ycbcr.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\trained_model\ycbcr_net.pkl'))
ycbcr.eval().cuda()
patch = patch_model()
patch.load_state_dict(torch.load(r'C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\trained_model\patch.pkl'))
patch.eval().cuda()

# path = r'E:\database\nuaa\data\test\1\0014_01_06_03_86.jpg'
path = r'C:\Users\Neticle\Desktop\testpic\1_17.jpg'
pil_img = Image.open(path).convert('RGB')
t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])
tr = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(96),
    transforms.Resize((256, 256)),
    transforms.ToTensor()])
img = t(pil_img)
img.unsqueeze_(dim=0)
img = Variable(img).cuda()
imge = tr(pil_img)
imge.unsqueeze_(dim=0)
imge = Variable(imge).cuda()

rgb_out = rgb(img)
rgb_out = functional.softmax(rgb_out)
rgb_out = rgb_out.cpu()
rgb_out = rgb_out.detach().numpy()
#rgb_result = int(numpy.argmax(rgb_out.cpu().detach().numpy()))
hsv_out = hsv(img)
hsv_out = functional.softmax(hsv_out)
hsv_out = hsv_out.cpu()
hsv_out = hsv_out.detach().numpy()
#hsv_result = int(numpy.argmax(hsv_out.cpu().detach().numpy()))
ycbcr_out = ycbcr(img)
ycbcr_out = functional.softmax(ycbcr_out)
ycbcr_out = ycbcr_out.cpu()
ycbcr_out = ycbcr_out.detach().numpy()
#ycbcr_result = int(numpy.argmax(ycbcr_out.cpu().detach().numpy()))
patch_out = patch(imge)
patch_out = functional.softmax(patch_out)
patch_out = patch_out.cpu()
patch_out = patch_out.detach().numpy()
#patch_result = int(numpy.argmax(patch_out.cpu().detach().numpy()))

#out = out.detach().numpy()
#ind = int(numpy.argmax(out))
# ind = out[0, 0]+out[0, 1]
print(rgb_out, hsv_out, ycbcr_out, patch_out)
#print(rgb_result, hsv_result, ycbcr_result, patch_result)


clf = joblib.load(r"C:\Users\Neticle\Desktop\bishedaima\Fusing Multiple\trained_model\svm_model.m")
test_X = [[rgb_out[0, 0], hsv_out[0, 0], ycbcr_out[0, 0], patch_out[0, 0]]]
print(clf.predict(test_X))
