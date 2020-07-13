"""
Includes classes and functions used to create 
a causal functional model for a given graph strucutre
"""

import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import torch as torch
from torch import nn
import copy

torch.manual_seed(123)

class CFMDataset(Dataset):
    """
    Implementation of pytorch dataset class
    to conveniently apply transforms, batchsizes, etc.
    """
    def __init__(self, features, labels, transform=None):

        self.all_features = features
        self.all_labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.all_features[idx], dtype=torch.double)
        label = torch.tensor(self.all_labels[idx], dtype=torch.double)
        
        if self.transform:
            features = self.transform(features)

        return features, label

def get_dataloader(data, feature_cols, label_cols, batch_size=1):
    """ 
    Returns pytorch dataloader from given dataframe
    """
    df = data.dropna(axis=0, how='any')
    dataset = CFMDataset(features=df[feature_cols].to_numpy(), 
                        labels=df[label_cols].to_numpy())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
    return dataloader

def func_block(dim_in, depth, activation=None):
    """
    Returns a MLP as specified by args
    """
    block = torch.nn.Sequential()
    for i in range(int(depth)-1):
        block.add_module('fc_' + str(i), nn.Linear(dim_in, dim_in))
        if activation is not None:
            block.add_module('activation_' + str(i), activation)
    block.add_module('fc_' + str(i+1), nn.Linear(dim_in, 1))  
    return block.double()

def xavier_init(m):  
    """
    Applies Xavier/Glorot initialization
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def model_graph(graph, data, val_split, batch_size, depth, activation=None, optimizer='SGD'):
    """
    Extends the graph by adding models, dataloaders, 
    and optimizers to nodes with inputs and returns it.

    TBD: Add support for substituting models through 
    user defined functions (in case of missing data).
    Add more optimizers.
    """
    modeled_graph = copy.deepcopy(graph)
    for node in modeled_graph:
        if node['inputs'] is not None:
            node['trainloader'] = get_dataloader(data[:val_split], feature_cols=node['inputs'],
                                    label_cols=[node['name']], batch_size=batch_size)
            node['valloader'] = get_dataloader(data[val_split:], feature_cols=node['inputs'],
                                    label_cols=[node['name']], batch_size=batch_size)                        
            node['model'] = func_block(len(node['inputs']), depth, activation)
            if optimizer == 'Adam':
                node['optimizer'] = torch.optim.Adam(node['model'].parameters())
            elif optimizer == 'SGD':
                node['optimizer'] = torch.optim.SGD(node['model'].parameters(), momentum=0.9, lr=0.001, weight_decay=0.00005)
            else: 
                print('Error: Choosen optimizer not implemented -> Default Adam is used')
                node['optimizer'] = torch.optim.Adam(node['model'].parameters())
    return modeled_graph