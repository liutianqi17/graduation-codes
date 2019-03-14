from torch import nn, optim

class cnn_model(nn.Module):
    def __init__(self, in_dim, n_class):
        super(cnn_model,self).__init__()
        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=50, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(150),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=150, out_channels=200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(250),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(2250, 1000),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.Linear(1000, 400),
            nn.ReLU(True),
            nn.BatchNorm1d(400),
            nn.Linear(400, n_class)
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