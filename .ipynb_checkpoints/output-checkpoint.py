from torch.serialization import load
from model import *
from resnet18 import Resnet18

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

model = NetBN()

model.load_state_dict(torch.load('modelfile/mnist_cnnbn.pt'))
print(model)
# for name in model.state_dict():
#     print(model.state_dict()[name].shape, name)