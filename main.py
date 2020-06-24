
"""
Create random dataset and a causal graph.
Append models and training data to graph nodes.
Train models seperatly. 
Use full graph to perform inference. 

TBD: Add normalization/standardization for data
Systematic testing. Add support for gml graphs etc. 
("compatibility" to causal inference packages)
"""

import numpy as np 
import pandas as pd
import torch.nn as nn
import functional as f
import model as m

def get_default_data(length=200):
    # Independent variables 
    x11 = np.random.rand(length)*2 - 1
    x12 = np.random.rand(length)*2 - 1
    x13 = np.random.rand(length)*2 - 1

    # Latent variables
    l21 = x12*0.4 + x13*0.6

    # Dependent variables
    d31 = x11*0.6 + l21*0.4

    df = pd.DataFrame({'NSEAMS':x11, 'POWER':x12, 'SPEED':x13, 'XSEC':l21, 'F':d31})
    return df



data = get_default_data(100)
split = 80

graph = [
    {
        'name': 'NSEAMS',
        'inputs': None,
    },
    {
        'name': 'POWER',
        'inputs': None,
    },
    {
        'name': 'SPEED',
        'inputs': None,
    },
    {
        'name': 'XSEC',
        'inputs': ['POWER', 'SPEED'],
    },
    {
        'name': 'F',
        'inputs': ['NSEAMS', 'XSEC'],
    }
]

modeled_graph = m.model_graph(graph, data, split, 1, 3)
trained_graph = f.train_graph(modeled_graph, 2, nn.L1Loss())           
pred = f.inference(trained_graph, data[['NSEAMS', 'POWER', 'SPEED']], 
                    data[['XSEC', 'F']], ['MAE', 'MSE'])
