# -*- coding: utf-8 -*-
# @Time : 2023/3/21 14:51
# @Author : 
# @Email : aaron.weihuang@gmail.com
# @File : model.py
# @Project : MultiQuant
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):        # 普通Block简单完成两次卷积操作
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x                                            # 普通Block的shortcut为直连，不需要升维下采样

        x = F.relu(self.bn1(self.conv1(x)))       # 完成一次卷积
        x = self.bn2(self.conv2(x))                             # 第二次卷积不加relu激活函数

        x += identity                                           # 两路相加
        return F.relu(x)



class SpecialBlock(nn.Module):                                  # 特殊Block完成两次卷积操作，以及一次升维下采样
    def __init__(self, in_channel, out_channel, stride):        # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
        super(SpecialBlock, self).__init__()

        self.pre1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=True)
        self.pre2 = nn.BatchNorm2d(out_channel)

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.pre2(self.pre1(x))                       # 调用change_channel对输入修改，为后面相加做变换准备
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))                             # 完成残差部分的卷积
        x += identity
        return F.relu(x) 



class Resnet18(nn.Module):

    def __init__(self, num_channels=3):
        super(Resnet18, self).__init__()
        # input feature operation
        self.conv1 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        # block 1
        self.block1_1 = CommonBlock(64, 64, 1)
        self.block1_2 = CommonBlock(64, 64, 1)
        # block 2
        self.block2_1 = SpecialBlock(64, 128, [2, 1])
        self.block2_2 = CommonBlock(128, 128, 1)
        # block 3
        self.block3_1 = SpecialBlock(128, 256, [2, 1])
        self.block3_2 = CommonBlock(256, 256, 1)
        # block 4
        self.block4_1 = SpecialBlock(256, 512, [2, 1])
        self.block4_2 = CommonBlock(512, 512, 1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):
        # input feature operation
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2, 1)
        # block 1
        x = self.block1_1(x)
        x = self.block1_2(x)
        #block 2
        x = self.block2_1(x)
        x = self.block2_2(x)
        #block 3
        x = self.block3_1(x)
        x = self.block3_2(x)
        #block 4
        x = self.block4_1(x)
        x = self.block4_2(x)

        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x