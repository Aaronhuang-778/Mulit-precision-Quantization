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

ratios = [0.75, 0.25]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_results = []



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
                # print('Test set: Full Model Accuracy: {:.2f}%'.format(100. * correct / length))
                return result, 100. * correct / length
            
    else:
                # print('Test set: Mask Model Accuracy: {:.2f}%'.format(100. * correct / length))
                return result, 100. * correct / length

def MaskInference(model_name, original_model, test_loader, key='conv', ratio=1, method=0):
    params = original_model.state_dict()
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
    
    model = ptcv_get_model(model_name, pretrained=True)
    model.load_state_dict(param_new)
    model.eval()
    return NormalInference(model, test_loader, key, ratio, True)


def SensitiveTest(model_name, original_model, file_path):
    
    draws = []
   
    with open(file_path,'a') as f:
        for ratio in ratios:
            f.write('---------------------------------------------------------\n')
            f.write("mask ratio : " + str(ratio) +";\n")
            f.write("layer name" + "\t" + "KL divergency" + "\t" + "accuracy\n")
            layer = -1
            for name, param in original_model.named_parameters():
                if ('conv' in name and 'weight' in name and 'bn' not in name) or ('conv' not in name and 'weight' in name and 'bn' not in name):
                    layer += 1
                    print("Test: ratio-" + str(ratio) + "; layer-" + name)
                    dl_results = []
                    accuracys = []
                    original_accuracys = []
                    for itr in range(20):
                        if model_name == "inceptionv3":
                                test = get_data(299)
                        else: test = get_data(224)
                        
                        original_results, accuracy = NormalInference(original_model, test)
                        original_accuracys.append(accuracy) 
                        mask_results, accuracy1 = MaskInference(model_name, original_model, test, name, ratio)
                        accuracys.append(accuracy1)
                        for i in range(len(original_results)):
                            original_outputs = original_results[i][0].detach().numpy()
                            mask_outputs = mask_results[i][0].detach().numpy()
                            dl_results.append(stats.entropy(original_outputs, mask_outputs))
                    mean_kl = np.mean(dl_results)
                    push_list = [ratios.index(ratio), layer, mean_kl]
                    print("KL divergency: " + str(mean_kl))
                    total_results.append(push_list)
                    accuracy = np.mean(accuracys)
                    print("accuracy : " + str(accuracy) + "\n")
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
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))

    transform_test = transforms.Compose([transforms.Resize(size), transforms.RandomCrop(size),transforms.ToTensor(), normalize])
    testset = datasets.ImageFolder(root='../../../../data/ILSVRC2012_img_val', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True)

    test = []
    for i, (data, target) in enumerate(test_loader, 1):
        a = [i, data, target]
        test.append(a)
        if i == 20:
            break
    
    return test
       
    
if __name__ == '__main__':
    
    net_list = ['resnet18', 'resnet50', 'resnet101', 'inceptionv3', 'mobilenet_w1']
    
    
    for model_name in net_list:
        model = ptcv_get_model(model_name, pretrained=True)
        print("-------------------Successfully load quantized model %s---------------" % model_name)
        model.eval()
        
        file_path = model_name + '_imagenet_full_mask_ratio.txt'
        SensitiveTest(model_name, model, file_path)
