from torch import nn, optim

class cnn_model(nn.Module):
    def __init__(self, in_dim, n_class):
        super(cnn_model,self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        )
        self.relu = nn.ReLU(True)
        self.downsaple1 = nn.Sequential(
            #nn.BatchNorm2d(in_dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.downsaple2 = nn.Sequential(
            #nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.downsaple3 = nn.Sequential(
            #nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class)
        )

    def forward(self, x):
        res = x
        out = self.conv1(x)
        res = self.downsaple1(res)
        out += res
        out = self.relu(out)
        res = out
        out = self.conv2(out)
        res = self.downsaple2(res)
        out += res
        out = self.relu(out)
        res = out
        out = self.conv3(out)
        res = self.downsaple3(res)
        out += res
        out = self.relu(out)
        res = out
        out = self.conv4(out)
        out += res
        out = self.relu(out)
        res = out
        out = self.conv4(out)
        out += res
        out = self.relu(out)
        res = out
        out = self.conv4(out)
        out += res
        out = self.relu(out)
        res = out
        out = self.conv4(out)
        out += res
        out = self.relu(out)
        res = out
        out = self.conv4(out)
        out += res
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out