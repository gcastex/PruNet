"""
Disclaimer:

This code is based on code by Peter Ruch.
See his prunhild repository: https://github.com/gfrogat/prunhild
Snippets of code also borrowed from Arun Joseph pruning code.
https://github.com/00arun00/Pruning-Pytorch/blob/master/prune.py
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms

from mtcnn_pytorch_master.src import detect_faces, show_bboxes
from PIL import Image
from LightCNN_master.load_imglist import ImageList


def load_training_set(batch_size = 20):
    root_path = '/home/ubuntu/Project_Insight/Data/GOT/'
    train_list = root_path+'train.txt'
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=root_path, fileList=train_list,
        transform=transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        pin_memory=True)
    return train_loader


def load_validation_set(batch_size = 20):
    root_path = '/home/ubuntu/Project_Insight/Data/GOT/'
    val_list = root_path+'val.txt'
    val_loader = torch.utils.data.DataLoader(
        ImageList(root=root_path, fileList=val_list,
        transform=transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,

        pin_memory=True)
    return val_loader

def load_test_set(batch_size = 20):
    root_path = '/home/ubuntu/Project_Insight/Data/GOT/'
    test_list = root_path+'test.txt'
    test_loader = torch.utils.data.DataLoader(
        ImageList(root=root_path, fileList=test_list,
        transform=transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        #num_workers=args.workers,
        pin_memory=True)
    return test_loader


def setup_dataloaders(kwargs, dataset_to_use):
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if dataset_to_use == 'GOT':
        train_loader = load_training_set()
        train_loader_eval = load_validation_set()
        test_loader_eval = load_test_set()
    else:
        datafolder = "~/data/torch"
        dataset = (datasets.FashionMNIST)#if dataset_to_use == "FashionMNIST" else datasets.MNIST)
        train_loader = torch.utils.data.DataLoader(
            dataset(datafolder, train=True, download=True, transform=mnist_transform),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
            )
        train_loader_eval = torch.utils.data.DataLoader(
            dataset(datafolder, train=True, download=True, transform=mnist_transform),
            batch_size=args.batch_size_eval,
            shuffle=True,
            **kwargs
        )
        test_loader_eval = torch.utils.data.DataLoader(
            dataset(datafolder, train=False, transform=mnist_transform),
            batch_size=args.batch_size_eval,
            shuffle=True,
            **kwargs
        )
    return train_loader, train_loader_eval, test_loader_eval


def train(
    model,
    device,
    data_loaders,
    optimizer,
    epoch,
    use_mask = False,
    prune=False,
    cut_ratio = .2
):
    prune_interval = 400
    print_interval = 200
    eval_interval = 200
    train_loader, train_loader_eval, test_loader_eval = data_loaders
    logs = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
	if sum(sum(torch.isnan(output))).cpu().detach().numpy() > 0:
		print(batch_idx)
        loss = F.cross_entropy(output, target)
	if (loss.cpu().detach().numpy()) > 1000: # For debugging purposes
		print(loss)
		print('=============')
		break
        loss.backward()
        if use_mask or prune:
            # Make sure the pruned parameters aren't updated
            for index,params in enumerate(model.parameters()):
                mask = model.prune_mask[index]
                params.grad.data[mask]=0
        optimizer.step()
        # Pruning
        if prune:
            if (batch_idx % prune_interval == 0) and (batch_idx > 0):
                print('-')
                print(batch_idx)
                model.prune_model(cut_ratio)
        if batch_idx % print_interval == 0:
            print(
                  "[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Frac Zeros: {:.2f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    model.get_prune_frac()
                )
            )
        if batch_idx % eval_interval == 0:
            ratio_zero = model.get_prune_frac()
            acc_train = evaluate(model, device, train_loader_eval, training = True)
            acc_test = evaluate(model, device, test_loader_eval)
            logs.append((epoch, batch_idx, ratio_zero, acc_train, acc_test))
    return logs


def evaluate(model, device, data_loader, training=False):
    fold = "Train" if training else "Test"
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    print(
        "[Eval] {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f})".format(
            fold, test_loss, correct, len(data_loader.dataset), accuracy
        )
    )
    return accuracy


def get_seed(i,j):
    seeds = [[10,22000,13,154,65],[832,120,1294,138,4567], [43, 616, 94, 1732, 7253]]
    return int(seeds[i][j])


def set_seed(seed):
    torch.manual_seed(seed)


def prune_and_train(prune_frac = .9, seed_model = 1, max_epoch = 200, dir_out = '', save_all = False, cmdline_args=None):
    if prune_frac >=1. :
	print('prune_frac must be in [0,1[.')
	return 0
    model_to_use = 'LCNN'
    batch_size_eval= 512
    learning_rate= 5e-3
    momentum= 0.8
    cutoff_ratio= 0.15
    cut_ratio = .2
    if model_to_use == 'LCNN':
    	prune_interval = 10
        prune_train = False
        prune_epoch = True    
    else:
        prune_interval = 0
        prune_train = True
        prune_epoch = False
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    data_loaders = setup_dataloaders(kwargs,dataset_to_use = "GOT")
    model = TestNN().to(device) #LotteryLeNet().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )
    logs_prune = []
    print("Pruning Start")
    set_seed(seed_model)
    model.apply(init_weights)
    epoch = 1 
    while model.get_prune_frac() < prune_frac:
        if prune_epoch: 
            if (epoch % prune_interval == 0):
                print('-epoch-')
                print(epoch)
                model.prune_model(cut_ratio)
		if save_all:
    			prune_percent = int(np.round(model.get_prune_frac()*100))
    			savefile_model = dir_out+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'.pt'
    			torch.save(model.state_dict(), savefile_model)
    			savefile_mask = dir_out+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'_mask.sav'
    			torch.save(model.prune_mask, savefile_mask)
        logs_prune += train(
            model, device, data_loaders, optimizer, epoch, prune=prune_train, use_mask = True, cut_ratio = cut_ratio
        )
	epoch += 1
	if epoch >= max_epoch:
		break
    # Train the model after final pruning
    for i in range(2*prune_interval):
        logs_prune += train(
            model, device, data_loaders, optimizer, epoch, prune=prune_train, use_mask = True, cut_ratio = cut_ratio
        )
	epoch += 1
    print("\n\n\n")
    prune_percent = int(np.round(model.get_prune_frac()*100))
    savefile_model = dir_out+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'.pt'
    torch.save(model.state_dict(), savefile_model)
    savefile_logs = dir_out+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'_logs.sav'
    savefile_mask = dir_out+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'_mask.sav'
    torch.save(logs_prune, savefile_logs)
    torch.save(model.prune_mask, savefile_mask)
    return logs_prune


def retrain(epochs=10, prune_percent = 0, seed_model=1, seed_baseline =0 ,cmdline_args=None):
    dir_sav = 'saves/'
    model_to_use = 'LCNN'
    batch_size_eval= 512
    learning_rate= 1e-2
    momentum= 0.8
    cutoff_ratio= 0.15
    cut_ratio = .2
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    data_loaders = setup_dataloaders(kwargs,dataset_to_use = "GOT")
    if model_to_use == 'LCNN':
        prune_interval = 10
        prune_train = False
        prune_epoch = True
    else:
        prune_interval = 0
        prune_train = True
        prune_epoch = False
    ###########
    # Load model
    savefile_model = dir_sav+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'.pt'
    model = TestNN().to(device)
    model.load_state_dict(torch.load(savefile_model))
    savefile_mask = dir_sav+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'_mask.sav'
    model.prune_mask = torch.load(savefile_mask)
    n_trials =2 
    nvar_log = 5
    logs_retrain = torch.empty(n_trials, epochs, nvar_log, dtype=torch.double)
    for i_test in range(n_trials):
        logs_i = []
        if model_to_use == 'LCNN':
            model_retrain = LightCNN().to(device)
        else:
            model_retrain = PruneNN().to(device)
        if i_test <1:
            model_retrain.reinit(seed_model, seed_model, n_layer = i_test)
        else:
            set_seed(seed_baseline)
            model_retrain.apply(init_weights)
        optimizer_retrain = optim.SGD(
            model_retrain.parameters(), lr=learning_rate, momentum=momentum
        )
        print("Retraining Start -- "+"str(i_test)")
        # Restore mask
        model_retrain.prune_mask = model.prune_mask
        for epoch in range(1, epochs + 1):
            logs_i += train(
                model_retrain,
                device,
                data_loaders,
                optimizer_retrain,
                epoch,
                use_mask = True,
                prune=False
            )
        print("\n\n\n")
        logs_retrain[i_test,:,:] = torch.DoubleTensor(logs_i)
    savefile_retrain = dir_sav+'model'+model_to_use+'_prune_'+str(prune_percent)+'seed_'+str(seed_model)+'_retrain.sav'
    torch.save(logs_retrain, savefile_retrain)
    return logs_retrain
