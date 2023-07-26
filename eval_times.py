#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse

import os

import torchvision
#import json
import pandas as pd
import time
#import wandb
import utils
import eval
import models

def main(args):

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    dict_log = dict()
    dict_log["checkpoint"] = []
    dict_log["model"] = []
    dict_log["eval_time_testset_seconds"] = []
    dict_log["acc"] = []

    for cl in classes:
        dict_log[cl] = []
        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    #dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=utils.transform(model_class ,"train"))
    #trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    batch_size = 4
    num_workers = 4
    
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=utils.transform('resnet50', "test"))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
      
 

    if args.checkpoints_dir:
        checkpoints = args.checkpoints_dir
    else:
        checkpoints = args.models
    checkpoints_ls = os.listdir(checkpoints)
    for cp in checkpoints_ls:
        
    # train from scratch or load existing checkpoint
        path_to_load_from = os.path.join(checkpoints, cp)
        
        if args.checkpoints_dir:
            print(path_to_load_from)
            checkpoint = torch.load(path_to_load_from)
            
            #initial_epochs = checkpoint["epoch"]
            #print(initial_epochs)
            model_class = checkpoint["model_class"]
            

            
            #optimizer_name = checkpoint["optimizer_name"]
            #parameters = checkpoint["parameters"]
        
            model = models.model_choice(model_class)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
        else:
            model = torch.load(cp)
            model_class = cp.split('.')
        #loss_fn = checkpoint["loss_fn"]
        #batch_size = checkpoint["batch_size"]
        #num_workers = checkpoint["num_workers"]

        print("loading model for pruning", model_class)
        # log accuracy and time for unpruned model
        st = time.time()
        acc, class_acc = eval.evaluate(classes, testloader, device, model)
        eval_time = st - time.time()
        
        dict_log["checkpoint"].append(cp)
        dict_log["model"].append(model_class)
        dict_log["eval_time_testset_seconds"].append(eval_time)
        dict_log["acc"].append(acc)
        
        for cl in classes:
            dict_log[cl].append(class_acc[cl])
        
      
   

            

                
 
    #model_class_savename = os.path.join(checkpoints, model_class)
    #os.mkdirs(model_class_savename)        
    df = pd.DataFrame.from_dict( dict_log)
    df.to_csv(os.path.join(checkpoints, "eval.csv"))
    print(os.path.join(checkpoints, "eval.csv"))
        
        

 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
        # either path to config file or checkpoint required
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--checkpoints_dir", type = str, help= "directory with checkpoints")
    
    # optional
    group.add_argument(
        "--models", type=str, help="directory with saved models")
    

    args = parser.parse_args()


    main(args)
