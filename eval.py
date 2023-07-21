#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

def evaluate(classes, testloader, device, model):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    correct = 0
    total = 0
    
    # again no gradients needed
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # ??? batch_size = imgs.shape[0]
            # ???outputs = model_2(imgs.view(batch_size, -1))
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            total += labels.shape[0]
            correct += int((predictions == labels).sum())
            
    total_acc = correct/total
    print("Total Accuracy: ", total_acc )
    # print accuracy for each class
    class_acc_dict = dict()
    
    for classname, correct_count in correct_pred.items():
        accuracy =  float(correct_count) / total_pred[classname]
        class_acc_dict[classname] = accuracy
        print(f'Accuracy for class: {classname:5s} is {accuracy:.4f}')
    

    return(total_acc, class_acc_dict)
    