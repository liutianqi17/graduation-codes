import torch
from torch import nn, optim
from random import randint
from config import *


class cnn_model(nn.Module):
    def __init__(self, in_dim, n_class):
        super(cnn_model,self).__init__()
        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(81920, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, n_class)
        )

    def forward(self, x):
        patch_x = randint(0, img_size-patch_size)
        patch_y = randint(0, img_size-patch_size)
        out1 = x[:, :, patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
        out1 = self.conv1(out1)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)
        out1 = self.conv4(out1)
        out1 = out1.view(out1.size(0), -1)
        patch_x = randint(0, img_size - patch_size)
        patch_y = randint(0, img_size - patch_size)
        out2 = x[:, :, patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        out2 = self.conv1(out2)
        out2 = self.conv2(out2)
        out2 = self.conv3(out2)
        out2 = self.conv4(out2)
        out2 = out1.view(out2.size(0), -1)
        patch_x = randint(0, img_size - patch_size)
        patch_y = randint(0, img_size - patch_size)
        out3 = x[:, :, patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        out3 = self.conv1(out3)
        out3 = self.conv2(out3)
        out3 = self.conv3(out3)
        out3 = self.conv4(out3)
        out3 = out1.view(out3.size(0), -1)
        patch_x = randint(0, img_size - patch_size)
        patch_y = randint(0, img_size - patch_size)
        out4 = x[:, :, patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        out4 = self.conv1(out4)
        out4 = self.conv2(out4)
        out4 = self.conv3(out4)
        out4 = self.conv4(out4)
        out4 = out1.view(out4.size(0), -1)
        patch_x = randint(0, img_size - patch_size)
        patch_y = randint(0, img_size - patch_size)
        out5 = x[:, :, patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        out5 = self.conv1(out5)
        out5 = self.conv2(out5)
        out5 = self.conv3(out5)
        out5 = self.conv4(out5)
        out5 = out1.view(out5.size(0), -1)
        out = torch.cat([out1, out2, out3, out4, out5], 1)
        out = self.fc(out)
        return out