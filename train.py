#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime

import numpy as np
import os
import torch
import torchvision
import json
import wandb
import pandas as pd
from tqdm import tqdm

import models
import utils
import eval


def main(args):
    #wandb.login()
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dict_log = dict()
    dict_log["epoch"] = []
    dict_log["loss"] = []
    dict_log["lr"] = []
    dict_log["acc"] = []

    
    
    for cl in classes:
        dict_log[cl] = []
        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # train from scratch or load existing checkpoint
    if args.load:
        print("loading model to continue training")
        path_to_load_from = args.load
        checkpoint = torch.load(path_to_load_from)
        
        initial_epochs = checkpoint["epoch"]
        print(initial_epochs)
        model_class = checkpoint["model_class"]
        
        optimizer_name = checkpoint["optimizer_name"]
        parameters = checkpoint["parameters"]
    
        model = models.model_choice(model_class)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        loss_fn = checkpoint["loss_fn"]
        batch_size = checkpoint["batch_size"]
        num_workers = checkpoint["num_workers"]
        
        
    else :
        with open(args.config_file, "r") as f:
            config = json.load(f)
        print("training new model with:")
        print(config)
        

        
        model_class = config["model"]
        model = models.model_choice(model_class)
        loss_fn = models.loss_fn_choice(config["loss_fn"])
        batch_size = config["batch_size"]
        n_epochs = config["n_epochs"]
        optimizer_name = config["optimizer_name"]
        initial_epochs = 0 
        parameters = config["parameters"]
        num_workers = config["num_workers"]
        model.to(device)
        #learning_rate = config["parameters"]["learning_rate"]
        
    if optimizer_name =="SGD":
        optimizer =torch.optim.SGD(model.parameters(), lr= parameters["learning_rate"], momentum=parameters["momentum"], weight_decay = parameters["weight_decay"])
            
    else: 
        raise NameError("unknown optimizer name in config file")
    if args.load:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(optimizer)
    
    total_epochs = initial_epochs
    
    if args.batch_size:
        batch_size = args.batch_size
    if args.epochs:
        n_epochs = args.epochs
    
    if args.num_workers:
        num_workers = args.num_workers
    
    

    # data 
    

    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=utils.transform(model_class ,"train"))
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=utils.transform(model_class, "test"))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
      
 

    
    
    # train

    print("training on ", device, " for ", n_epochs, " epochs")
    
    # the lr scheduler which everyone seems to use for resnet on cifar
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    """
    run = wandb.init(
            project = "EML_project",
            config = {
                    "learning_rate": scheduler.get_lr(),
                    "epochs": total_epochs + n_epochs ,
                    
                    }
            
            )
    """

    
    for epoch in range(n_epochs): 
        total_epochs += 1

        running_loss = 0.0
        for i, (imgs, labels) in enumerate(tqdm(trainloader), 0):
           
            imgs, labels = imgs.to(device), labels.to(device)
            
            # batch_Size = imgs.shape[0]
            # outputs = model_2(imgs.view(batch_size,-1))


            optimizer.zero_grad()

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
           
        
        print("Epoch: ", total_epochs , " loss: ", float(running_loss))
        

        
        if epoch < 200:
            scheduler.step()
        
        if epoch % 1 == 0:
            
            acc, class_acc = eval.evaluate(classes, testloader, device, model)
            #wandb.log({"accuracy": acc, "loss": running_loss})        
            print(acc)

            
            dict_log["acc"].append(acc)
            dict_log["loss"].append(running_loss)
            dict_log["lr"].append(scheduler.get_last_lr()[0])
            dict_log["epoch"].append(total_epochs)
                
            for cl in classes:
                dict_log[cl].append(class_acc[cl])
            
            
            
            
            # dir to save checkpoints
            if args.load:
                checkpoints_savedir = os.path.split(args.load)[0]
                print(checkpoints_savedir)
            else:
                checkpoints_savedir = model_class + "_" + config["name"] + "_" + str(config["version"])
            savedir_path = os.path.join("checkpoints", checkpoints_savedir)
            os.makedirs(savedir_path, exist_ok = True)
            
            # save model with timestamp
            d = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            savename =  os.path.join(savedir_path, "epoch" + str(total_epochs)+ "_" + d)
            
            
            torch.save({
                    'epoch': total_epochs,
                    'model_class': model_class,
                    'optimizer_name': optimizer_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'parameters': parameters,
                    'loss_fn': loss_fn,
                    'batch_size': batch_size,
                    'num_workers': num_workers
                    }
            , savename)   
            print("checkpoint saved as ", savename)
            
            if args.load:
                df_savename = "epoch" +str( initial_epochs+ n_epochs) + "from epoch" + str(initial_epochs) 
            else:
                df_savename = "epoch" +str( initial_epochs + n_epochs) 
            df = pd.DataFrame.from_dict( dict_log)
            df.to_csv(os.path.join(savedir_path, df_savename + ".csv"))
        
        df = pd.DataFrame.from_dict( dict_log)
        df.to_csv(os.path.join(savedir_path, df_savename + "_" + d +".csv"))
            

            
    print('training completed')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # either path to config file or checkpoint required
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--config_file", type=str, help="path to config yaml file")
    
    # optional
    group.add_argument(
        "--load", type=str, help="path where model to be loaded is stored"
    )
    
    parser.add_argument("--epochs", type = int, help= "number of epochs to train, if not from config")
    parser.add_argument("--batch_size", type = int, help= "batch size, if not from config")
    parser.add_argument("--num_workers", type = int, help= "number of workers, it not from config")
    

    args = parser.parse_args()
    
    if args.load and (args.epochs is None):
        parser.error("--load requires --epochs.")

    main(args)

   