import torch
from torch import nn as nn


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim, layers=(128, 64, 32), activation=nn.ReLU, dropout=None, bn=True):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        seqs = []
        last_layer = input_dim
        for layer in layers:
            seqs.append(nn.Linear(last_layer, layer))
            last_layer = layer
            if activation is not None:
                seqs.append(activation())
            if dropout is not None:
                seqs.append(nn.Dropout(p=dropout))
            if bn:
                seqs.append(nn.BatchNorm1d(layer))
        seqs.append(nn.Linear(last_layer, 1))
        self.net = nn.Sequential(*seqs)

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        return self.criterion(pred, target)


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=False, dropout=0.0):
        super(BasicBlock, self).__init__()
        blocks = [nn.Linear(input_dim, output_dim)]
        if bn:
            blocks.append(torch.nn.BatchNorm1d(output_dim))
        blocks.append(nn.ReLU())
        if dropout > 0.0:
            blocks.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block(x)
        return x


class NNClassifier(nn.Module):
    def __init__(self, layers, bn=False, dropout=0.0):
        super(NNClassifier, self).__init__()
        layer_seqs = []
        for i in range(len(layers) - 2):
            layer_seqs.append(BasicBlock(layers[i], layers[i + 1], bn=bn, dropout=dropout))
        layer_seqs.append(nn.Linear(layers[-2], layers[-1]))
        self.fc = nn.Sequential(
            *layer_seqs
        )

    def forward(self, x):
        x = self.fc(x)
        return x
