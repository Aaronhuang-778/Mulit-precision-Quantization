# -*- coding: utf-8 -*-
# @Time : 2023/4/6 17:18
# @Author : 
# @Email : aaron.weihuang@gmail.com
# @File : test.py
# @Project : code
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import tensor
from torchvision import datasets, transforms
from PIL import Image
import ssl
from scipy import stats
from resnet18 import Resnet18
from functools import partial
import copy

ratios = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0]

def Flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)

def Gauss_noise_matrix(matrix, sigma):
    
    mu = 0
    shape = matrix.shape
    noise_matrix = np.random.normal(mu, sigma, size=shape).astype(np.float32)
    matrix += noise_matrix
    
    return matrix

def NormalInference(model, test_loader, key='conv', ratio=0.2, mask=False):
    result = []
    correct = 0

    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        m = nn.Softmax(dim=1)
        outputs_softmax = m(output)
        result.append(outputs_softmax[0])
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if i == 1000:
            if not mask:
                print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / i))
                return result
            else:
                condition = 'Mask: key=' + str(key) + str(ratio)
                accuracy = 'Test set: Mask Model Accuracy: {:.2f}%\n'.format(100. * correct / i)
                return result, condition, accuracy

def MaskInference(model, test_loader, key='conv', ratio=1, method=1):
    original_weight = model.state_dict()[key].data.numpy()
    
    # method with random mask operation
    if method == 0:
        weight, unflatten = Flatten(original_weight)
        len = weight.shape[0]
        nums = np.ones(len)
        nums[:round(len*ratio)] = 0
        np.random.shuffle(nums)
        final_weight = nums * weight
        unflatten(final_weight)
    # mtehod with tiny gaussian disturbe    
    else:
        final_weight = Gauss_noise_matrix(original_weight, 0.02)

    model.state_dict()[key].data = torch.from_numpy(final_weight)

    return NormalInference(model, test_loader, key, ratio, True)


def SensitiveTest(model, test_loader, original_results):
    with open("sensitive/resnet_18_0_0.875",'a') as f:
        for name, param in model.named_parameters():
            if ('conv' in name and 'weight' in name) or ('fc' in name and 'weight' in name):
                f.write('---------------------------------------------------------\n')

                for ratio in ratios:
                    print("Test: ratio-" + str(ratio) + "; layer-" + name)
                    mask_model = copy.deepcopy(model)
                    mask_results, condition, accuracy = MaskInference(mask_model, test_loader, name, ratio)
                    f.write(condition + '\n')
                    f.write(accuracy + '\n')
                    dl_results = []
                    for i in range(len(original_results)):
                        original_outputs = original_results[i].detach().numpy()
                        mask_outputs = mask_results[i].detach().numpy()
                        dl_results.append(stats.entropy(original_outputs, mask_outputs))
                    mean_kl = np.mean(dl_results)

                    f.write("layer_name : " + name + ";  mask ratio : " + str(ratio) + ";\n")
                    f.write("KL divergency : " + str(mean_kl) + "\n")
                    f.write('---------------------------------------------------------\n')
                
# def SensitiveTest(model, test_loader, original_results):
#     name = "fc1.weight"
#     for ratio in ratios:
#         mask_model = copy.deepcopy(model)

#         mask_results = MaskInference(model, test_loader, name, ratio)

#         dl_results = []
#         for i in range(len(original_results)):
#             original_outputs = original_results[i].detach().numpy()
#             mask_outputs = mask_results[i].detach().numpy()
#             print(original_outputs)
#             print(mask_outputs)
#             dl_results.append(stats.entropy(original_outputs, mask_outputs))
#         mean_kl = np.mean(dl_results)
                
#         print("layer_name : " + name + ";  mask ratio : " + str(ratio) + ";")
#         print("KL divergency : " + str(mean_kl) + "\n")            

if __name__ == '__main__':

    model = Resnet18()
    load_quant_model_file = 'modelfile/resnet18_2.pt'
    model.load_state_dict(torch.load(load_quant_model_file, map_location='cpu'))
    print("Successfully load quantized model %s" % load_quant_model_file)
    model.eval()
    data_transform = {                      # 数据预处理
            "val": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        }

    test_data = datasets.CIFAR10(root='../../../data/', train=False, transform=data_transform["val"], download=False)
    test_loader = torch.utils.data.DataLoader(test_data, 1, False, num_workers=0)
    
    original_results = NormalInference(model, test_loader)
    SensitiveTest(model, test_loader, original_results)
