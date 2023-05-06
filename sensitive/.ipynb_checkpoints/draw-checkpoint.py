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
from functools import partial
import copy
from pytorchcv.model_provider import get_model as ptcv_get_model
from resnet18 import Resnet18

ratios = [1, 0.875, 0.75, 0.5, 0.25, 0.125]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_results = []
original_model = Resnet18()
original_model.load_state_dict(torch.load('../modelfile/resnet18_2.pt', map_location='cpu'))
print("-------------------Successfully load quantized model ---------------")
original_model.eval()
params = original_model.state_dict()

def Flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)

def Gauss_noise_matrix(matrix, sigma):
    
    mu = 0
    shape = matrix.shape
    noise_matrix = np.random.normal(mu, sigma, size=shape).astype(np.float32)
    matrix += noise_matrix
    
    return matrix

def NormalInference(model, test, key='conv', ratio=0.2, mask=False):
    result = []
    correct = 0
    length = len(test)
    for item in test:
        i = item[0]
        data = item[1]
        target = item[2]
        output = model(data)
        m = nn.Softmax(dim=1)
        outputs_softmax = m(output)
        result.append(outputs_softmax)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    if not mask:
                print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / length))
                return result, 100. * correct / length
            
    else:
                print('Test set: Mask Model Accuracy: {:.2f}%\n'.format(100. * correct / length))
                return result, 100. * correct / length

def MaskInference(test_loader, key='conv', ratio=1, method=0):
    
    param_new = {}
    for i in params.keys():
        param_new[i] = params[i]
    
    original_weight = original_model.state_dict()[key].data.numpy()
    # method with random mask operation
    if method == 0:
        weight, unflatten = Flatten(original_weight)
        lenth = weight.shape[0]
        nums = np.ones(lenth)
        nums[:round(lenth*ratio)] = 0
        np.random.shuffle(nums)
        final_weight = nums * weight
    # mtehod with tiny gaussian disturbe    
    else:
        final_weight = Gauss_noise_matrix(original_weight, 0.02)

    # model.state_dict()[key].data = torch.from_numpy(final_weight)
    param_new[key] = torch.from_numpy(unflatten(final_weight))
    
    model = Resnet18()
    model.load_state_dict(param_new)
    model.eval()
    return NormalInference(model, test_loader, key, ratio, True)


def SensitiveTest(test, original_results, file_path, original_accrucy):
    
    draws = []
   
    with open(file_path,'a') as f:
        f.write('Full precision accuracy is: ' + str(original_accrucy) + "\n")
        for ratio in ratios:
            f.write('---------------------------------------------------------\n')
            f.write("mask ratio : " + str(ratio) +";\n")
            f.write("layer name" + "\t" + "KL divergency" + "\t" + "accuracy\n")
            layer = -1
            for name, param in original_model.named_parameters():
                if ('conv' in name and 'weight' in name) or ('fc' in name and 'weight' in name):
                    layer += 1
                
                    print("Test: ratio-" + str(ratio) + "; layer-" + name)
                    mask_results, accuracy = MaskInference(test, name, ratio)
                    
                    dl_results = []
                    for i in range(len(original_results)):
                        original_outputs = original_results[i][0].detach().numpy()
                        mask_outputs = mask_results[i][0].detach().numpy()
                        dl_results.append(stats.entropy(original_outputs, mask_outputs))
                    mean_kl = np.mean(dl_results)
                    push_list = [ratios.index(ratio), layer, mean_kl]
                    total_results.append(push_list)
                    f.write(name + "\t" +  str(mean_kl) + "\t" + str(accuracy) + "\n")
                
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
def get_data(size):
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
    test_data = datasets.CIFAR10(root='../../../../data/', train=False, transform=data_transform["val"], download=False)
    test_loader = torch.utils.data.DataLoader(test_data, 1, True, num_workers=0)
    test = []
    for i, (data, target) in enumerate(test_loader, 1):
        a = [i, data, target]
        test.append(a)
        if i == 500:
            break
    
    return test
       
    
if __name__ == '__main__':
    
    
    test = get_data(224)
        
    file_path ='resnet18_cifar_full_mask_ratio.txt'
    original_results, accrucy= NormalInference(original_model, test)
    SensitiveTest(test, original_results, file_path, accrucy)
    print(total_results)
    

        