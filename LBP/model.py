from torch import nn, optim
from torchvision import models

class vgg_model(nn.Module):
    def __init__(self, in_dim, n_class):
        super(vgg_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, n_class)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class res_model(nn.Module):
    def __init__(self):
        super(res_model, self).__init__()
        self.res = fine_tune_model()

    def forward(self, x):
        out = self.res(x)
        return out


def fine_tune_model():
    model_ft = models.resnet18(pretrained=False)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(2048, 2)
    return model_ft
