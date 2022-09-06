import os
import random

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
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


def hyper_parameters():
    parameters = {
        'concat_nframes': 1,
        'train_ratio': 0.8,
        'seed': 0,
        'batch_size': 512,
        'num_epochs': 5,
        'learning_rate': 0.0001,
        'model_path': './model.ckpt',
        'data_root': './data/hw2/',
    }
    parameters['layers'] = [parameters['concat_nframes'] * 1, 256, 41]

    return parameters


def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X


def get_data(params):
    feat_dir = os.path.join(params['data_root'], './libriphone/feat')
    phone_path = os.path.join(params['data_root'], './libriphone')
    train_X, train_y = preprocess_data(split='train', feat_dir=feat_dir, phone_path=phone_path,
                                       concat_nframes=params['concat_nframes'], train_ratio=params['train_ratio'])
    val_X, val_y = preprocess_data(split='val', feat_dir=feat_dir, phone_path=phone_path,
                                   concat_nframes=params['concat_nframes'], train_ratio=params['train_ratio'])
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False)

    return train_loader, val_loader


def train(train_loader, val_loader=None, params={}, device='cpu'):
    model = Classifier(params['layers'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])

    for epoch in range(params['num_epochs']):
        train_acc = 0.0
        train_loss = 0.0

        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = features.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, labels)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()


def main():
    params = hyper_parameters()
    train_loader = get_data(params)
    train(train_loader, params=params)


if __name__ == '__main__':
    main()
