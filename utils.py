#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchvision.transforms as transforms

def transform(name, train_test):
    if name == "simple_example":
        return  transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if (name == "resnet_18") or (name == "resnet50") or (name == "mobilenet_v2"):
        if train_test == "train":
            return transforms.Compose([
                    transforms.Resize((224, 224)),   
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                    ])
        elif train_test == "test":
            return transforms.Compose([
                    transforms.Resize((224, 224)),   
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])




"""
# transforms found for mobilenetv2
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""