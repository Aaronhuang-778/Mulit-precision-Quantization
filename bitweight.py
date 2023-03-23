import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

from torch.serialization import load
from model import *

model= test_jit()
model.quantize(8)
model.load_state_dict(torch.load('modelfile/mnist_jit_int8.pt'))
state_dict = torch.load('modelfile/mnist_jit_int8.pt')
model.freeze()
print(model)
for name in model.state_dict():
    print(model.state_dict()[name], name)
# for i in model.children():
#     print(i)