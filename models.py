#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def model_choice(name):
    if name =="simple_example":
        model = Toy()
    elif name == "resnet_18":
        model = torchvision.models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
    elif name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
    elif name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(pretrained=True)
        num_classes = 10
        model.classifier[1] = torch.nn.Linear(1280, num_classes)

        
    else: 
        raise NameError("unknown model name in config file")
    return model

def loss_fn_choice(name):
    if name =="cross_entropy":
        l_fn = nn.CrossEntropyLoss()
    else: 
        raise NameError("unknown loss function name in config file")
    return l_fn


class Toy(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
