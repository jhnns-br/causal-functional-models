"""
Functions for training of and inference 
on the causal functional graph

TBD: algo for extrakting a graph from data 
--> e.g. assessing the "explainatory value" 
of a input_x to node_y
"""

import numpy as np 
import pandas as pd 
import torch as torch
from torch import nn
from sklearn import metrics
import copy

torch.manual_seed(123)

def train_graph(modeled_graph, max_epochs, criterion, early_stopping=True):
    """
    Train the models of the graph node using the 
    provided dataloaders and optimizers. 
    """
    trained_graph = copy.deepcopy(modeled_graph)
    print('\nTRAINING:')
    for node in trained_graph:
        if node['inputs'] is not None:
            print(' ------------------\t', node['name'], '\t------------------')
            min_loss = 1000.0
            min_loss_epoch = 0
            stop_ctr = 5

            for epoch in range(max_epochs):

                train_loss = 0.0
                stat_train_loss = 0.0 
                test_loss = 0.0
                stat_test_loss = 0.0

                node['model'].train()
                for _, (features, labels) in enumerate(node['trainloader']):
                    node['optimizer'].zero_grad()
                    out = node['model'](features)    
                    train_loss = criterion(out, labels)
                    train_loss.backward()
                    node['optimizer'].step()
                    stat_train_loss += train_loss.item()

                node['model'].eval()
                for _, (features, labels) in enumerate(node['valloader']):     
                    out = node['model'](features)  
                    test_loss = criterion(out, labels)
                    stat_test_loss += test_loss.item()
  
                print(' Ep', epoch, '\tTrain Loss:',
                    round(stat_train_loss/len(node['trainloader']),5), '\tVal. Loss:',
                    round(stat_test_loss/len(node['valloader']), 5))
    
                if stat_test_loss/len(node['valloader']) < min_loss: 
                    min_loss = stat_test_loss/len(node['valloader'])
                    min_loss_epoch = epoch
                if early_stopping and epoch - min_loss_epoch > stop_ctr:
                    break

    return trained_graph

def inference(trained_graph, df_inputs, df_labels=None, metric=['MAE']):
    """
        Function uses input values provided in df_inputs.
        If df_inputs containes values for node_x
        the values in df_inputs will be used as inputs for 
        subsequent node_x+1, not the predictions of node_x. 

        TBD: graph must be strucutred sequentially when created 
            --> implent so some sort of sorting algo
    """
    
    df_pred = pd.DataFrame()
    for node in trained_graph:
        if node['inputs'] is not None:
            X = torch.tensor(df_inputs[node['inputs']].to_numpy(), dtype=torch.double)
            y = []
            for x in X:
                y_pred = node['model'](x)
                y.append(y_pred.item())
            df_pred[node['name']] = y
            df_inputs.loc[:, node['name']] = y
    
    if df_labels is not None:
        print('\nTESTING:')
        for label in df_labels.columns:
            for m in metric:
                if m == 'MAE':
                    res = metrics.mean_absolute_error(df_labels[[label]].to_numpy(), 
                                                    df_pred[[label]].to_numpy())
                elif m == 'MSE':
                    res = metrics.mean_squared_error(df_labels[[label]].to_numpy(), 
                                                    df_pred[[label]].to_numpy())
                else:
                    res = metrics.mean_absolute_error(df_labels[[label]].to_numpy(), 
                                                    df_pred[[label]].to_numpy())
                    m = 'undefined metric --> defaults to MAE'


                print(' ', label, m + ':', "   \t", round(res, 5))
    
    return df_pred    