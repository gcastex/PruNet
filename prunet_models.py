"""
Disclaimer:

This code is based on codes by Peter Ruch, Arun Joseph and Alfred Xiang Wu.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == torch.nn.modules.conv.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class PruneNN(nn.Module):
    """Test and model masterclass
    """
    def __init__(self):
        super(PruneNN, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 3)
        self.initialize_mask()
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
        print('Pruning--------------')
        #for index,params in enumerate(self.parameters()):
        for index, (name, params) in enumerate(self.named_parameters()):
            if not 'bias' in name: # Do not prune biases
                p = cut_ratio+(1-cut_ratio)*torch.sum(self.prune_mask[index].view(-1)).double()/(self.prune_mask[index].view(-1)).size(0)
                p = p.detach().numpy()
                threshold = np.percentile(torch.abs(params.data.view(-1).cpu()).detach().numpy(), p*100)
                mask = torch.abs(params.data)<=threshold
                self.prune_mask[index] = np.logical_or(self.prune_mask[index],mask.cpu())
                params.data[mask]=0
                if verbose:
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

######### LightCNN implementation. Adapted from code by Alfred Xiang Wu.

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class LighCNN(PruneNN):
    """LightCNN, model by Alfred Xiang Wu
        """
    def __init__(self, num_classes=8):
        super(TestNN, self).__init__()
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
        self.initialize_mask()
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out
