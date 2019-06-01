"""
Disclaimer:

This code is based on code by Peter Ruch, see his prunhild repository
See: https://github.com/gfrogat/prunhild
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

#import prunhild

from config import parser
from utils import get_parameter_stats, print_parameter_stats



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# model.prune_model(.2,verbose=True)

class PruneNN(nn.Module):
    """Test and model masterclass
    """
    def __init__(self):
        super(PruneNN, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        #self.fc1 = nn.Linear(4, 5)
        #self.fc2 = nn.Linear(5, 5)
        #self.fc3 = nn.Linear(5, 3)
        #self.initialize_mask()
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def initialize_mask(self):
        self.prune_mask={}
        for index,params in enumerate(self.parameters()):
            self.prune_mask[index] = (params.data) < (params.data-1)
    def prune_model(self,cut_ratio, verbose = False):
        # Prune the model and removes cut_ratio weights for each layer.
        if not hasattr(self, 'prune_mask'):
            self.initialize_mask()
        nlayers2 = len(list(self.parameters())) #Number of layers*2 (weights+bias for each)
        for index,params in enumerate(self.parameters()):
            if index < nlayers2-2: # Do not prune output layer
                #newmask = True
                #print(len(self.prune_mask)>index)
                if len(self.prune_mask)>index:
                    p = cut_ratio+(1-cut_ratio)*torch.sum(self.prune_mask[index].view(-1)).double()/(self.prune_mask[index].view(-1)).size(0)
                    p = p.detach().numpy()
                #newmask = False
                else:
                    p = cut_ratio
                threshold = np.percentile(torch.abs(params.data.view(-1)).detach().numpy(), p*100)
                mask = torch.abs(params.data)<=threshold
                    #if newmask:
                    #self.prune_mask[index] = mask
                    #else:
                self.prune_mask[index] = np.logical_or(self.prune_mask[index],mask)
                #if verbose:
                    #For testing purposes
                    #print(threshold)
                    #print ("--params--")
                    #print(params)
                    #print ("-new params-")
                #Set pruned parameters to 0
                params.data[mask]=0
                if verbose:
                    #print(params)
                    print(p)
                    print("------")
                    print(mask)
                    print("=========")
    def reinit(self, seed1, seed2, n_layer=0):
        torch.manual_seed(seed1)
        self.apply(init_weights)
        torch.manual_seed(seed2)
        if n_layer > 0:
            # Layer 0 doesn't exist. Set n_layer=0 to reinitialize all layers using seed 1.
            init_weights(getattr(self, 'fc'+ str(n_layer)))
    def get_prune_frac(self):
        zeros = 0
        total = 0
        for index,params in enumerate(self.parameters()):
            zeros+=torch.sum(self.prune_mask[index].view(-1))
            total+=(self.prune_mask[index].view(-1)).size(0)
        return np.asscalar((zeros.double()/total).detach().numpy())


class TestNN(PruneNN):
    """LightCNN, model by Alfred Xiang Wu
        """
    def __init__(self):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)
        self.prune_mask={}
        for index,params in enumerate(self.parameters()):
            self.prune_mask[index] = (params.data) < (params.data-1)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out




###
"""
for index,params in enumerate(model.parameters()):
    newmask = True
    if len(model.prune_mask)>index:
        p = cut_ratio+(1-cut_ratio)*sum(model.prune_mask[index].view(-1)).double()/((model.prune_mask[index].view(-1)).size(0))
        p = p.detach().numpy()
        newmask = False
    else:
        p = cut_ratio
    threshold = np.percentile(torch.abs(params.data.view(-1)).detach().numpy(), p*100)
    mask = torch.abs(params.data)<=threshold
    if newmask:
        model.prune_mask[index] = mask
    else:
        model.prune_mask[index] = np.logical_or(model.prune_mask[index],mask)
"""
###

##
#model = LotteryLeNet()
#model.reinit(1,1,1)
#print(model.fc1.weight.data[:1,:16])
##

def setup_dataloaders(args, kwargs):
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = (
        datasets.FashionMNIST if args.dataset == "FashionMNIST" else datasets.MNIST
    )
    train_loader = torch.utils.data.DataLoader(
        dataset(args.datafolder, train=True, download=True, transform=mnist_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    train_loader_eval = torch.utils.data.DataLoader(
        dataset(args.datafolder, train=True, download=True, transform=mnist_transform),
        batch_size=args.batch_size_eval,
        shuffle=True,
        **kwargs
    )
    test_loader_eval = torch.utils.data.DataLoader(
        dataset(args.datafolder, train=False, transform=mnist_transform),
        batch_size=args.batch_size_eval,
        shuffle=True,
        **kwargs
    )
    return train_loader, train_loader_eval, test_loader_eval


def train(
    args,
    model,
    device,
    data_loaders,
    optimizer,
          #pruner,
    epoch,
    use_mask = False,
    prune=False#,
          #prune_online=True,
):
    print("Test--------------------------")
    prune_interval = 50
    print_interval = 200
    eval_interval = 200
    cut_ratio = .2
    train_loader, train_loader_eval, test_loader_eval = data_loaders
    logs = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if use_mask or prune:
            # Make sure the pruned parameters aren't updated
            for index,params in enumerate(model.parameters()):
                mask = model.prune_mask[index]
                params.grad.data[mask]=0
        optimizer.step()
        # Pruning
        if prune: # and prune_online is True:
            #if batch_idx >= args.prune_delay or epoch > 1:
            if batch_idx % prune_interval == 0:
                model.prune_model(cut_ratio)
                #pruner.prune()
        # ---- Pruning Instrumentation End ---- #
        # -------------------------------------- #
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
            #parameter_stats = get_parameter_stats(model)
            #print_parameter_stats(parameter_stats)
            #_, _, ratio_zero = parameter_stats
            ratio_zero = model.get_prune_frac()
            acc_train = evaluate(args, model, device, train_loader_eval)
            acc_test = evaluate(args, model, device, test_loader_eval)
            logs.append((epoch, batch_idx, ratio_zero, acc_train, acc_test))
        #print("Hello here!")
        #break
    return logs


def evaluate(args, model, device, data_loader):
    fold = "Train" if data_loader.dataset.train is True else "Test"
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
    seeds = [[10,22000,13,154,65],[832,120,1294,138,4567]]
    return int(seeds[i][j])


def set_seed(i,j):
    torch.manual_seed(get_seed(i,j))


def run_test(cmdline_args=None):
    args = parser.parse_args(cmdline_args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    seed_list = 0
    seed_number = 1
    data_loaders = setup_dataloaders(args, kwargs)
    model = TestNN().to(device) #LotteryLeNet().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
    # --------------------------- #
    # --- Pruning Setup Start --- #
    #cutoff = prunhild.cutoff.LocalRatioCutoff(args.cutoff_ratio)
    # don't prune the final bias weights
    #params = list(model.parameters())[:-1]
    #pruner = prunhild.pruner.CutoffPruner(params, cutoff, prune_online=True)
    # ---- Pruning Setup End ---- #
    # --------------------------- #
    logs_prune = []
    print("Pruning Start")
    #torch.manual_seed(args.seed_dataloader)
    # Initialize weights
    set_seed(seed_list,seed_number)
    model.apply(init_weights)
    # print(model.fc1.weight.data[:1,:16])
    for epoch in range(1, args.epochs + 1):
        #print("+=+=+=+=+=+=")
        logs_prune += train(
            args, model, device, data_loaders, optimizer, epoch, prune=True
        )
    print("\n\n\n")
    # -------------------------------------- #
    # --- Pruning Weight Resetting Start --- #
    # we want to demonstrate here how to export and load the state of a pruner
    # i.e. a actual sparse model or LotteryTicket that we want to train from
    # scratch now. Make sure that the architecture and the parameters match!
    #pruner_state = pruner.state_dict()
    # reset seed for initializing with the same weights
    #torch.manual_seed(args.seed_retrain)
    #set_seed(seed_list,seed_number)
    n_trials = 5
    logs_retrain = torch.empty(n_trials, len(logs_prune),len(logs_prune[0]), dtype=torch.double)
    for i_test in range(n_trials):
        logs_i = []
        model_retrain = TestNN().to(device) #LotteryLeNet().to(device)
        if i_test <4:
            seed1 = get_seed(seed_list,seed_number)
            seed2 = get_seed(1,i_test)
            model_retrain.reinit(seed1, seed2, n_layer = i_test)
        else:
            set_seed(seed_list,seed_number+1)
            model_retrain.apply(init_weights)
        optimizer_retrain = optim.SGD(
            model_retrain.parameters(), lr=args.learning_rate, momentum=args.momentum
        )
        #cutoff_retrain = prunhild.cutoff.LocalRatioCutoff(args.cutoff_ratio)
        #params_retrain = list(model_retrain.parameters())[:-1]
        #pruner_retrain = prunhild.pruner.CutoffPruner(params_retrain, cutoff_retrain)
        # now we load the state dictionary with the prune-masks that were used last
        # for pruning the model.
        #pruner_retrain.load_state_dict(pruner_state)
        # calling prune with `update_state=False` will simply apply the last prune_mask
        # stored in state
        #pruner_retrain.prune(update_state=False)
        print("Retraining Start")
        # Restore mask
        model_retrain.prune_mask = model.prune_mask
        torch.manual_seed(args.seed_dataloader_retrain)
        for epoch in range(1, args.epochs + 1):
            logs_i += train(
                args,
                model_retrain,
                device,
                data_loaders,
                optimizer_retrain,
                            #pruner_retrain,
                epoch,
                use_mask = True,
                prune=False
            )
        print("\n\n\n")
        logs_retrain[i_test,:,:] = torch.DoubleTensor(logs_i)
    # ---- Pruning Weight Resetting End ---- #
    # -------------------------------------- #
    return logs_prune, logs_retrain


if __name__ == "__main__":
    run_test()
    
#>>> from my_test import run_test
#>>> logs_prune, logs_retrain = run_test([])

    column_names = ["Epoch", "Iteration", "Ratio Weights Zero", "Accuracy Train", "Accuracy Test"]
    import pandas as pd
    import matplotlib.pyplot as plt
    df_prune = pd.DataFrame(logs_prune, columns=column_names)
    df_prune.set_index("Iteration", inplace=True)
    logs_rbase = logs_retrain[0].data.numpy()
    df_rbase = pd.DataFrame(logs_rbase, columns=column_names)
    df_rbase.set_index("Iteration", inplace=True)
    logs_r1 = logs_retrain[1].data.numpy()
    df_r1 = pd.DataFrame(logs_r1, columns=column_names)
    df_r1.set_index("Iteration", inplace=True)
    logs_r2 = logs_retrain[2].data.numpy()
    df_r2 = pd.DataFrame(logs_r2, columns=column_names)
    df_r2.set_index("Iteration", inplace=True)
    logs_r3 = logs_retrain[3].data.numpy()
    df_r3 = pd.DataFrame(logs_r3, columns=column_names)
    df_r3.set_index("Iteration", inplace=True)
    logs_rnd = logs_retrain[4].data.numpy()
    df_rnd = pd.DataFrame(logs_rnd, columns=column_names)
    df_rnd.set_index("Iteration", inplace=True)

    df_merged = df_prune.merge(df_rbase, left_index=True, right_index=True, suffixes=('', ' (rbase)'))
    df_merged = df_merged.merge(df_r1, left_index=True, right_index=True, suffixes=('', ' (r1)'))
    df_merged = df_merged.merge(df_r2, left_index=True, right_index=True, suffixes=('', ' (r2)'))
    df_merged = df_merged.merge(df_r3, left_index=True, right_index=True, suffixes=('', ' (r3)'))
    df_merged = df_merged.merge(df_rnd, left_index=True, right_index=True, suffixes=(' (prune)', ' (rnd)'))
    df_merged[["Ratio Weights Zero (prune)", "Ratio Weights Zero (rbase)","Ratio Weights Zero (r1)","Ratio Weights Zero (r2)","Ratio Weights Zero (r3)","Ratio Weights Zero (rnd)"]].plot(ylim=(0, 1), title="Ratio Weights Zero", figsize=(12, 7))
    plt.show()
    
    ax1 = df_merged[["Accuracy Train (prune)",  "Accuracy Train (rbase)", "Accuracy Train (rnd)"]].plot(figsize=(12, 7))
    
    
    ax1 = df_merged[["Accuracy Train (prune)", "Accuracy Train (rbase)", "Accuracy Test (rbase)", "Accuracy Train (r1)", "Accuracy Test (r1)", "Accuracy Train (r2)", "Accuracy Test (r2)", "Accuracy Train (r3)", "Accuracy Test (r3)"]].plot(figsize=(12, 7))
    ax1 = df_merged[["Accuracy Train (prune)",  "Accuracy Train (rbase)", "Accuracy Train (r1)", "Accuracy Train (r2)", "Accuracy Train (r3)", "Accuracy Train (rnd)"]].plot(figsize=(12, 7))
    ax1.set_ylim(.7,1.)
    plt.show()
    ax1 = df_merged[["Accuracy Test (prune)", "Accuracy Test (rbase)", "Accuracy Test (r1)", "Accuracy Test (r2)", "Accuracy Test (r3)", "Accuracy Test (rnd)"]].plot(figsize=(12, 7))
    ax1.set_ylim(.7,1.)
    plt.show()

    plt.hist(model.fc1.weight.view(-1).detach().numpy(), bins=30)
    #df_retrain = pd.DataFrame(logs_retrain, columns=column_names)
    #df_retrain.set_index("Iteration", inplace=True)

    #df_merged = df_prune.merge(df_retrain, left_index=True, right_index=True, suffixes=(' (prune)', ' (retrain)'))
