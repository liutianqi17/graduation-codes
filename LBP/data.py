import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from config import *

def loadtraindata():
    path = r"E:\database\CBSR-Antispoofing\LBP\train"
    # path = r"E:\database\nuaa\LBP\train"
    train_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            transforms.Resize((300, 300)),
                            transforms.CenterCrop(256),
                            transforms.ToTensor()
                            # ,transforms.RandomSizedCrop(64)
                            ])
                            )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return train_loader, train_dataset

def loadtestdata():
    path = r"E:\database\CBSR-Antispoofing\LBP\test"
    # path = r"E:\database\nuaa\LBP\test"
    test_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            transforms.Resize((300, 300)),
                            transforms.CenterCrop(256),
                            transforms.ToTensor()
                            # ,transforms.RandomSizedCrop(64)
                            ])
                            )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    return test_loader, test_dataset
