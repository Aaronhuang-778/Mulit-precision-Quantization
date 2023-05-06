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


# normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
# transform_test = transforms.Compose([
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
#         normalize
    # ])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
test_data = datasets.CIFAR10(root='../../../../data/', train=False, transform=data_transform["val"], download=False)

test_dataloader = torch.utils.data.DataLoader(test_data, test_batch_size, False, num_workers=0)


net_list = ['resnet18', 'resnet50', 'resnet101', 'vgg16', 'mobilenet_w1', 'efficientnet_b0', 'inceptionv3']

net = ptcv_get_model("resnet18", pretrained=True)
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
