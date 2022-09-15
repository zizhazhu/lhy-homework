import csv

import numpy as np
import torch
from torch.utils.data import Dataset


class COVID19Dataset(Dataset):
    """ Dataset for loading and preprocessing the COVID19 dataset """
    def __init__(self, path, mode='train', selection=None):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            # remove header and id
            data = np.array(data[1:])[:, 1:].astype(float)

        if selection is None:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            feats = []
            for i in range(len(selection)):
                if selection[i]:
                    feats.append(i)

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            else:
                assert mode == 'dev'
                indices = [i for i in range(len(data)) if i % 10 == 0]

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        if selection is None:
            self.data[:, 40:] = \
                (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
                / self.data[:, 40:].std(dim=0, keepdim=True)
        else:
            self.data[:, :] = \
                (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True)) \
                / self.data[:, :].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
