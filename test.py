from torch.serialization import load
from model import *
from resnet18 import Resnet18

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


# for name, param in mymodel.named_parameters():
#     print(name)
#     print(param.data)
def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output= model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))
    
def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))


batch_size = 64
quantize = True
dataset = "mnist"
model = test_jit()
load_quant_model_file = 'modelfile/mnist_jit_int8.pt'
if quantize:
    model.quantize(num_bits=8)

model.load_state_dict(torch.load(load_quant_model_file))
print("Successfully load quantized model %s" % load_quant_model_file)


model.eval()

test_loadet = None
if dataset == "mnist":
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
        )
else:
    data_transform = {                      # 数据预处理
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
    # 加载数据集，指定训练或测试数据，指定于处理方式
    test_data = datasets.CIFAR10(root='../../../data/', train=False, transform=data_transform["val"], download=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, False, num_workers=0)
if quantize:
    model.freeze()
    for name in model.state_dict():
        print(model.state_dict()[name], name)
    quantize_inference(model, test_loader)
else: 
    full_inference(model, test_loader)