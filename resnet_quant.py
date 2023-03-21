from torch.serialization import load
from resnet18 import Resnet18

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data)
        if i % 500 == 0:
            break
    print('direct quantization finish')


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.4f}%\n'.format(100. * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.4f}%\n'.format(100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 64

    # load_model_file = None

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
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size, False, num_workers=0)


    model = Resnet18()
    model.load_state_dict(torch.load('modelfile/resnet18_2.pt', map_location='cpu'))
    save_file = "modelfile/resnet18"

    num_bits = 8
    model.quantize(num_bits=num_bits)
    model.eval()
    print('Quantization bit: %d' % num_bits)
    save_file = save_file + "_int" + str(num_bits) + ".pt"
    

    direct_quantize(model, train_dataloader)
    model.freeze()
    torch.save(model.state_dict(), save_file)

    quantize_inference(model, test_dataloader)

    



    