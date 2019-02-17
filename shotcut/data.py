import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from config import *

def loadtraindata():
    path = r"E:\database\nuaa\data\train"
    train_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            #transforms.Resize((224, 224)),
                            #transforms.CenterCrop(224),
                            transforms.RandomSizedCrop(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
                            )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return train_loader, train_dataset

def loadtestdata():
    path = r"E:\database\nuaa\data\test"
    test_dataset = datasets.ImageFolder(path,
                            transform=transforms.Compose([
                            # transforms.Resize((224, 224)),
                            # transforms.CenterCrop(224),
                            transforms.RandomSizedCrop(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
                            )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    return test_loader, test_dataset