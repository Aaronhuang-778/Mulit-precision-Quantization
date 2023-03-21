from torch.serialization import load
from model import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

# model = NetBN()
# model.load_state_dict(torch.load('modelfile/mnist_cnnbn.pt', map_location='cpu'))

# model_8 = NetBN()
# model_8.quantize(num_bits=8)
# model_8.load_state_dict(torch.load('modelfile/mnist_cnnbn_int8.pt', map_location='cpu'))

# model_4 = NetBN()
# model_4.quantize(num_bits=4)

# model_4.load_state_dict(torch.load('modelfile/mnist_cnnbn_int4.pt', map_location='cpu'))

# print("-------------------------------FP 32-------------------------------")
# for name, parameters in model.named_parameters():#打印出每一层的参数的大小
#        print(name, ':', parameters.size())

# print("-------------------------------IN 4-------------------------------")
# for name, parameters in model_4.named_parameters():#打印出每一层的参数的大小
#        print(name, ':', parameters.size())

# print("-------------------------------IN 8-------------------------------")
# for name, parameters in model_8.named_parameters():#打印出每一层的参数的大小
#        print(name, ':', parameters.size())

# for name, param in mymodel.named_parameters():
#     print(name)
#     print(param.data)
batch_size = 64

def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.4f}%\n'.format(100. * correct / len(test_loader.dataset)))
    
    
model = QNetBN()
load_quant_model_file = 'modelfile/mnist_cnnbn_int8.pt'
model.quantize(num_bits=8)

model.load_state_dict(torch.load(load_quant_model_file))
print("Successfully load quantized model %s" % load_quant_model_file)

# for name in model.state_dict():
#     if not model.state_dict()[name].equal(torch.load(load_quant_model_file)[name]):
#         print(model.state_dict()[name], name)
print(model.state_dict()['qfc.M'])
print(torch.load(load_quant_model_file)['qfc.M'])
# torch.save(model.state_dict(), 'modelfile/mnist_cnnbn_int8_2.pt')

# cont = torch.load('modelfile/mnist_cnnbn_int8_2.pt')
# for i in cont.keys():
#     print(cont[i], i, cont[i].shape)

# test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#         batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
#     )
# quantize_inference(model, test_loader)