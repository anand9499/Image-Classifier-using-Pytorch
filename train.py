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

parser = argparse.ArgumentParser(description='train network')

parser.add_argument('data_dir', action="store", default="./flowers")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./classifier.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=1024)

args = parser.parse_args()

loc = args.data_dir
path = args.save_dir
lr = args.learning_rate
model_type = args.arch
dropout = args.dropout
hidden_units = args.hidden_units
device = args.gpu
epochs = args.epochs
print_every = 40
trainloader, validloader, testloader = helper.load_data(loc)

model, optimizer, criteria = helper.buildnn(model_type, device, hidden_units, dropout,lr)

helper.model_train(model, lr, criteria, optimizer, trainloader, validloader, print_every, epochs, device)

helper.save_checkpoint(path, model_type, hidden_units, dropout, lr, epochs)
