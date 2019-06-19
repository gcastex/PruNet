
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def same_model(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
    	if p1.data.ne(p2.data).sum() > 0:
        	return False
    return True


def disp_first_p(model):
    for index,params in enumerate(model.parameters()):
	print(params[0,:,:,:])
	break
    return params[0,:,:,:]



def print_layer_type(m):
    print(type(m))

def print_test_gc(m):
    print(type(m) == torch.nn.modules.conv.Conv2d) 
