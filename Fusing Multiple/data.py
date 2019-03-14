import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from config import *

def loadtraindata(patch = False, batch = batch_size):
    # path = r"E:\database\CBSR-Antispoofing\processed\train"
    path = r"E:\database\nuaa\data\train"
    train_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor()]))
    if patch:
        train_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.RandomCrop(96),
                            transforms.Resize((256, 256)),
                            transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch,
                                              shuffle=True, num_workers=0)
    return train_loader, train_dataset

def loadtestdata(patch = False, batch = batch_size):
    # path = r"E:\database\CBSR-Antispoofing\processed\test"
    path = r"E:\database\nuaa\data\test"
    test_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor()]))
    if patch:
        test_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.RandomCrop(96),
                            transforms.Resize((256, 256)),
                            transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch,
                                              shuffle=False, num_workers=0)
    return test_loader, test_dataset