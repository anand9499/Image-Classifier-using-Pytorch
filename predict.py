import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import helper

parser = argparse.ArgumentParser(description='predict class')
parser.add_argument('input_img', default='aipnd-project/flowers/test/27/image_06864.jpg', nargs='*', action="store", type = str)
parser.add_argument('checkpoint', default='/home/workspace/apind-project/checkpoint.pth', nargs='*', action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()

image_path = args.input_img
device = args.gpu
input_img = args.input_img
path = args.checkpoint
top_num = args.top_k

helper.load_modelcheckpoint(path)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
probabilities, classes, flowers = predict(image_path, model)
print(probabilities)
print(classes)
print(flowers)
