#!/usr/bin/env python

import torch
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# creates a conditional INN object using the FrEIA package with num_layers many layers,
# hidden neurons given by sub_net_size, dimension and dimension_condition specifying the dim of x/y respectively
# returns a nn.module object
def create_INN(num_layers, sub_net_size,dimension=5,dimension_condition=5):
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))
    nodes = [InputNode(dimension, name='input')]
    cond = ConditionNode(dimension_condition, name='condition')
    for k in range(num_layers):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.4},
                          conditions = cond,
                          name=F'coupling_{k}'))
    nodes.append(OutputNode(nodes[-1], name='output'))

    model = ReversibleGraphNet(nodes + [cond], verbose=False).to(device)
    return model

# trains an epoch of the INN
# given optimizer, the model and the data_loader
# training is done via maximum likelihood loss (conv_comb = 0 in SNF language)
# returns mean loss

def train_inn_epoch(optimizer, model, epoch_data_loader):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)

        loss = 0       
        invs, jac_inv = model(x, c = y, rev = True)

        l5 = 0.5 * torch.sum(invs**2, dim=1) - jac_inv
        loss += (torch.sum(l5) / cur_batch_size)

        optimizer.zero_grad()
        loss.backward()
           
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss



