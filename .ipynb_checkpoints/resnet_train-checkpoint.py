# -*- coding: utf-8 -*-
# @Time : 2023/3/13 15:02
# @Author : 
# @Email : aaron.weihuang@gmail.com
# @File : train.py
# @Project : MultiQuant
from model import *
from resnet18 import Resnet18
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import os
import os.path as osp


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    epochs = 30
    save_model = True


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    data_transform = {                      # 数据预处理
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #R,G,B每层的归一化用到的均值和方差
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
    # 加载数据集，指定训练或测试数据，指定于处理方式
    train_data = datasets.CIFAR10(root='../../../data/', train=True, transform=data_transform["train"], download=False)
    test_data = datasets.CIFAR10(root='../../../data/', train=False, transform=data_transform["val"], download=False)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size, True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_data, test_batch_size, False, num_workers=0)

    model = Resnet18().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, test_dataloader)

    if save_model:
        if not osp.exists('modelfile'):
            os.makedirs('modelfile')

        torch.save(model.state_dict(), 'modelfile/resnet18_2.pt')