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

from module import *

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

class QCommonBlock(nn.Module):
    def __init__(self, block, num_bits=8):        
        super(QCommonBlock, self).__init__()
        self.qconv1 =  QConvBNRelu(block.conv1, block.bn1, qi=False, qo=True, num_bits=num_bits)
        self.qconv2 =  QConvBN(block.conv2, block.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu = QRelu(qi=False, num_bits=num_bits)
    
    def forward(self, x):
        identity = x                                           
        x = self.qconv1(x)     
        x = self.qconv2(x)                        
        x += identity                                           
        return self.qrelu(x)
    
    def freeze(self, qi):
        self.qi = qi
        self.qconv1.freeze(qi=self.qi)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu.freeze(qi=self.qconv2.qo)
    
    def quantize_inference(self, qx):
        identity = qx
        q_x = self.qconv1.quantize_inference(qx)
        q_x = self.qconv2.quantize_inference(q_x)
        q_x += identity
        q_x = self.qrelu.quantize_inference(q_x)
        return q_x


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

class QSpecialBlock(nn.Module):                                  
    def __init__(self, block, num_bits=8):        
        super(QSpecialBlock, self).__init__()
        self.qpre = QConvBN(block.pre1, block.pre2, qi=False, qo=True, num_bits=num_bits)
        self.qconv1 =  QConvBNRelu(block.conv1, block.bn1, qi=False, qo=True, num_bits=num_bits)
        self.qconv2 =  QConvBN(block.conv2, block.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu = QRelu(qi=False, num_bits=num_bits)

    def forward(self, x):
        identity = self.qpre(x)                                           
        x = self.qconv1(x)     
        x = self.qconv2(x)                        
        x += identity                                           
        return self.qrelu(x)
    
    def freeze(self, qi):
        self.qi = qi
        self.qpre.freeze(qi=self.qi)
        self.qconv1.freeze(qi=self.qpre.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu.freeze(qi=self.qconv2.qo)
    
    def quantize_inference(self, qx):
        identity = self.qpre.quantize_inference(qx)
        q_x = self.qconv1.quantize_inference(qx)
        q_x = self.qconv2.quantize_inference(q_x)
        q_x += identity
        q_x = self.qrelu.quantize_inference(q_x)
        return q_x


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

    def quantize(self, num_bits=8):
        self.qconv1 = QConvBNRelu(self.conv1, self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=3, stride=2, padding=1)

        self.qblock1_1 = QCommonBlock(self.block1_1, num_bits)
        self.qblock1_2 = QCommonBlock(self.block1_2, num_bits)
        self.qblock2_1 = QSpecialBlock(self.block2_1, num_bits)
        self.qblock2_2 = QCommonBlock(self.block2_2, num_bits)
        self.qblock3_1 = QSpecialBlock(self.block3_1, num_bits)
        self.qblock3_2 = QCommonBlock(self.block3_2, num_bits)
        self.qblock4_1 = QSpecialBlock(self.block4_1, num_bits)
        self.qblock4_2 = QCommonBlock(self.block4_2, num_bits)

        self.qadaptive_avg_pool2d = QAdaptiveAvgPooling2d(output_size=(1, 1))
        self.qfc1 = QLinear(self.fc1, qi=False, qo=True, num_bits=num_bits)
        self.qrelu = QRelu(qi=False, num_bits=num_bits)
        self.qfc2 = QLinear(self.fc2, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qmaxpool2d_1(x)

        x = self.qblock1_1.forward(x)
        x = self.qblock1_2.forward(x)
        x = self.qblock2_1.forward(x)
        x = self.qblock2_2.forward(x)
        x = self.qblock3_1.forward(x)
        x = self.qblock3_2.forward(x)
        x = self.qblock4_1.forward(x)
        x = self.qblock4_2.forward(x)

        x = self.qadaptive_avg_pool2d(x)
        x = x.reshape(x.shape[0], -1)
        x = self.qfc1(x)
        x = self.qrelu(x)
        x = self.qfc2(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qmaxpool2d_1.freeze(qi=self.qconv1.qo)

        self.qblock1_1.freeze(qi=self.qconv1.qo)
        self.qblock1_2.freeze(qi=self.qblock1_1.qconv2.qo)
        self.qblock2_1.freeze(qi=self.qblock1_2.qconv2.qo)
        self.qblock2_2.freeze(qi=self.qblock2_1.qconv2.qo)
        self.qblock3_1.freeze(qi=self.qblock2_2.qconv2.qo)
        self.qblock3_2.freeze(qi=self.qblock3_1.qconv2.qo)
        self.qblock4_1.freeze(qi=self.qblock3_2.qconv2.qo)
        self.qblock4_2.freeze(qi=self.qblock4_1.qconv2.qo)

        self.qadaptive_avg_pool2d.freeze(qi=self.qblock4_2.qconv2.qo)
        self.qfc1.freeze(qi=self.qblock4_2.qconv2.qo)
        self.qrelu.freeze(qi=self.qblock4_2.qconv2.qo)
        self.qfc2.freeze(qi=self.qblock4_2.qconv2.qo)


    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        
        qx = self.qblock1_1.quantize_inference(qx)
        qx = self.qblock1_2.quantize_inference(qx)
        qx = self.qblock2_1.quantize_inference(qx)
        qx = self.qblock2_2.quantize_inference(qx)
        qx = self.qblock3_1.quantize_inference(qx)
        qx = self.qblock3_2.quantize_inference(qx)
        qx = self.qblock4_1.quantize_inference(qx)
        qx = self.qblock4_2.quantize_inference(qx)

        qx = self.qadaptive_avg_pool2d.quantize_inference(qx)
        qx = qx.reshape(x.shape[0], -1)
        qx = self.qfc1.quantize_inference(qx)
        qx = self.qrelu.quantize_inference(qx)
        qx = self.qfc2.quantize_inference(qx)
        
        out = self.qfc2.qo.dequantize_tensor(qx)

        return out, qx