import torch
from torch.nn.utils import prune
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


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def prune_model(model, conv2d_prune_amount):
    parameters_to_prune = []

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
          
            parameters_to_prune.append((module, "weight"))
    #print(parameters_to_prune)

    prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount,) 
    return model


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def main(args):
    steps = [ 0.6, 0.7, 0.8]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dict_log = dict()
    dict_log["model"] = []
    dict_log["step"] = []
    dict_log["sparsity"] = []
    dict_log["eval_time_testset_seconds"] = []
    dict_log["acc"] = []

    for cl in classes:
        dict_log[cl] = []
        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    #dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=utils.transform(model_class ,"train"))
    #trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    batch_size = 64
    num_workers = 12
    
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=utils.transform('resnet50', "test"))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
      
 

    
    checkpoints = args.checkpoints_dir
    os.mkdir(os.path.join(checkpoints, "pruned_models"))
    for cp in os.listdir(checkpoints):
        
    # train from scratch or load existing checkpoint
        path_to_load_from = os.path.join(checkpoints, cp)
        print(path_to_load_from)
        checkpoint = torch.load(path_to_load_from)
        
        initial_epochs = checkpoint["epoch"]
        #print(initial_epochs)
        model_class = checkpoint["model_class"]
        
        model_class_savename = os.path.join(checkpoints, "pruned_models", model_class)
        os.mkdir(model_class_savename)
        print("loading model for pruning", model_class)
        
        optimizer_name = checkpoint["optimizer_name"]
        parameters = checkpoint["parameters"]
    
        model = models.model_choice(model_class)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        loss_fn = checkpoint["loss_fn"]
        #batch_size = checkpoint["batch_size"]
        #num_workers = checkpoint["num_workers"]
        
        # log accuracy and time for unpruned model
        st = time.time()
        acc, class_acc = eval.evaluate(classes, testloader, device, model)
        eval_time = st - time.time()
        
        dict_log["model"].append(model_class)
        dict_log["step"].append(0)
        dict_log["sparsity"].append(0)
        dict_log["eval_time_testset_seconds"].append(eval_time)
        dict_log["acc"].append(acc)
        
        for cl in classes:
            dict_log[cl].append(class_acc[cl])
        
      
        for step in steps:
            model = models.model_choice(model_class)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            print(step)
            # prune
            model_pruned = prune_model(model, step)
            # remove weights
            model_pruned = remove_parameters(model_pruned)
            
            # calculate and log sparsity
            sparsity_pruned = measure_global_sparsity(model_pruned)
            
            
            # eval and log time and acc
            
            st = time.time()
            acc, class_acc = eval.evaluate(classes, testloader, device, model_pruned)
            eval_time = st - time.time()
            
            dict_log["model"].append(model_class)
            dict_log["step"].append(step)
            dict_log["sparsity"].append(sparsity_pruned)
            dict_log["eval_time_testset_seconds"].append(eval_time)
            dict_log["acc"].append(acc)
           
            for cl in classes:
                dict_log[cl].append(class_acc[cl])
                
            savename = os.path.join(model_class_savename, cp +"_pruned_"+str(int(step*10)))
            print(savename)
            # save pruned checkpoint
            torch.save({
                    'epoch': initial_epochs,
                    'model_class': model_class,
                    'optimizer_name': optimizer_name,
                    'model_state_dict': model_pruned.state_dict(),
                    'optimizer_state_dict': checkpoint['optimizer_state_dict'],
                    'parameters': parameters,
                    'loss_fn': loss_fn,
                    'batch_size': batch_size,
                    'num_workers': num_workers
                    }
            , savename)  
            
        df = pd.DataFrame.from_dict( dict_log)
        df.to_csv(os.path.join(model_class_savename, cp +"_pruned.csv"))
        
        

 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # either path to config file or checkpoint required
    '''
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--config_file", type=str, help="path to config yaml file")
    
    # optional
    group.add_argument(
        "--load", type=str, help="path where model to be loaded is stored"
    )
    
    parser.add_argument("--epochs", type = int, help= "number of epochs to train, if not from config")
    parser.add_argument("--batch_size", type = int, help= "batch size, if not from config")
    '''
    parser.add_argument("--checkpoints_dir", type = str, help= "directory with checkpoints of models to prune", required = True)
    

    args = parser.parse_args()
    
    #if args.load and (args.epochs is None):
     #   parser.error("--load requires --epochs.")

    main(args)
