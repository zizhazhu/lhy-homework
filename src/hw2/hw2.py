import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y:
            self.label = y
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, layers):
        super(Classifier, self).__init__()
        layer_seqs = []
        for i in range(len(layers) - 2):
            layer_seqs.append(BasicBlock(layers[i], layers[i+1]))
        layer_seqs.append(nn.Linear(layers[-2], layers[-1]))
        self.fc = nn.Sequential(
            *layer_seqs
        )

    def forward(self, x):
        x = self.fc(x)
        return x
