from torch.serialization import load
from model import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import time


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
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))
    

def test(num_bits):
    batch_size = 64
    using_bn = True
    load_quant_model_file = None
    # load_model_file = None

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    t1 = time.time()
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    model = test_jit()
    model.load_state_dict(torch.load('modelfile/mnist_jit.pt', map_location='cpu'))
    save_file = "modelfile/mnist_jit"


    model.eval()
    full_inference(model, test_loader)
    t2 = time.time()
    # model.quantize(num_bits=num_bits)
    # model.eval()
    # print('Quantization bit: %d' % num_bits)
    # save_file = save_file + "_int" + str(num_bits) + ".pt"

    # if load_quant_model_file is not None:
    #     model.load_state_dict(torch.load(load_quant_model_file))
    #     print("Successfully load quantized model %s" % load_quant_model_file)
    
    # direct_quantize(model, train_loader)
    # # torch.save(model.state_dict(), save_file)

    # model.freeze()

    # quantize_inference(model, test_loader)

    print("\naverage inference time is : {:.4f} ms".format( (t2 - t1) * 1000 / 10000 ))
    print(t2 - t1)

if __name__ == "__main__":
    test(8)

    



    