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

def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        print("========================================")
        print(output, pred, target, pred.eq(target.view_as(pred)).sum().item())
        return
    print('\nTest set: Quant Model Accuracy: {:.4f}%\n'.format(100. * correct / len(test_loader.dataset)))
    

batch_size = 1
dataset = "cifar10"
model = Resnet18()
load_quant_model_file = 'modelfile/resnet18_int8.pt'
model.quantize(num_bits=8)
model.eval()

model.load_state_dict(torch.load(load_quant_model_file))
print("Successfully load quantized model %s" % load_quant_model_file)

model.freeze()
# for name in model.state_dict():
#     if '.qi' in name:
#         print(model.state_dict()[name], name)

# data_transform = {                      # 数据预处理
#     "val": transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
# }
# # 加载数据集，指定训练或测试数据，指定于处理方式
# test_data = datasets.CIFAR10(root='../../../data/', train=False, transform=data_transform["val"], download=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size, False, num_workers=0)

# quantize_inference(model, test_loader)