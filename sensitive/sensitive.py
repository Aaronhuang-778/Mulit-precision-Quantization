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

ratios = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    for i, (data, target) in enumerate(test, 1):
        output = model(data)
        m = nn.Softmax(dim=1)
        outputs_softmax = m(output)
        result.append(outputs_softmax[0])
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if i == 30: break

    if not mask:
                print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / 30))
                return result, 100. * correct / 30
    else:
                condition = 'Mask: key= ' + str(key) + "------" + str(ratio)
                accuracy = 'Test set: Mask Model Accuracy: {:.2f}%\n'.format(100. * correct / 30)
                print(accuracy)
                return result, condition, accuracy

def MaskInference(model, test_loader, key='conv', ratio=1, method=0):
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


def SensitiveTest(model, test, original_results, file_path, original_accrucy):
   
    with open(file_path,'a') as f:
        f.write('Full precision accuracy is: ' + str(original_accrucy) + "\n")
        for name, param in model.named_parameters():
            if ('conv' in name and 'weight' in name) or ('fc' in name and 'weight' in name):
                f.write('---------------------------------------------------------\n')

                for ratio in ratios:
                    print("Test: ratio-" + str(ratio) + "; layer-" + name)
                    mask_model = copy.deepcopy(model)
                    mask_model.eval()
                    mask_results, condition, accuracy = MaskInference(mask_model, test, name, ratio)
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
def get_data(size):
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))

    transform_test = transforms.Compose([transforms.Resize(size), transforms.RandomCrop(size),transforms.ToTensor(), normalize])
    testset = datasets.ImageFolder(root='../../../../data/ILSVRC2012_img_val', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True)
    return test_loader
       
    
if __name__ == '__main__':
    
    net_list = ['vgg16', 'mobilenet_w1', 'efficientnet_b0', 'inceptionv3']
    
    test = get_data(224)
    
    for model_name in net_list:
        model = ptcv_get_model(model_name, pretrained=True)
        print("-------------------Successfully load quantized model %s---------------" % model_name)
        
        size = 0
        if model_name == "inceptionv3":
            test = get_data(299)
        model.eval()
        
        file_path = model_name + 'full_mask_ratio.txt'
        original_results, accrucy= NormalInference(model, test)
        SensitiveTest(model, test, original_results, file_path, accrucy)
