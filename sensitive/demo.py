from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.autograd import Variable

# x = Variable(torch.randn(1, 3, 224, 224))

# for name, param in net.named_parameters():
#     print(name)
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

num_label = 200
normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))

transform_test = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.ToTensor(), normalize])

# normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
# transform_test = transforms.Compose([
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
#         normalize
    # ])
testset = datasets.ImageFolder(root='../../../../data/ILSVRC2012_img_val', transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True)

net_list = ['resnet18', 'resnet50', 'resnet101', 'vgg16', 'mobilenet_w1', 'efficientnet_b0', 'inceptionv3']

net = ptcv_get_model("resnet50", pretrained=True)
net.eval()
for name, param in net.named_parameters():
    print(name, param.shape)
result = []
correct = 0

for i, (data, target) in enumerate(test_loader, 1):
    print(data.shape)
    output = net(data)
    m = nn.Softmax(dim=1)
    outputs_softmax = m(output)
    result.append(outputs_softmax[0])
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    print(i, target, pred)
    if i == 5:
            print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / i))
            break
