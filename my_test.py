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


def run_test2():
    seed_list = [501, 99, 807, 738, 372]
    seed_list_retrain = [333, 456, 295, 948, 994]
    for i in range(5):
        prune_and_train(prune_frac = .99, seed_model = seed_list[i], save_all = True, dir_out = 'saves/')
        #retrain(epochs = 50, prune_percent = 83, seed_model = seed_list[i], seed_baseline = seed_list_retrain[i])

def run_test3():
    prune_list = [36, 49,59,69,74,79,83,87, 89, 91, 93, 94, 96, 97, 98, 99]
    seed_list = [501, 99, 807, 738, 372]
    seed_list_retrain = [333, 456, 295, 948, 994]
    for j in range(len(prune_list)):
    	for i in range(5):
		retrain(epochs = 50, prune_percent = prune_list[j], seed_model = seed_list[i], seed_baseline = seed_list_retrain[i])

def run_test():
    seed_list = [501, 99, 807, 738, 372]
    seed_list_retrain = [333, 456, 295, 948, 994]
    for i in range(1):
    	prune_and_train(prune_frac = .8, seed_model = seed_list[i], dir_out = 'saves/') 
	retrain(epochs = 50, prune_percent = 83, seed_model = seed_list[i], seed_baseline = seed_list_retrain[i])
