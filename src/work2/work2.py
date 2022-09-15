import os
import math
import datetime
import random
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from util import get_device, set_rand_seed


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trail_id', type=str, default='test')
    parser.add_argument('--timestamp', action='store_true')
    parser.add_argument('mode', type=str)
    return parser


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


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=1, dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, query):
        x, _ = self.attention.forward(query, query, query, need_weights=False)
        return x


class PositionalEncoding(nn.Module):  # documentation code
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()  # new shortcut syntax
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # like 10x4
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # allows state-save

    def forward(self, x):
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)


class NNClassifier(nn.Module):
    def __init__(self, layers, bn=False, dropout=0.0):
        super(NNClassifier, self).__init__()
        layer_seqs = []
        for i in range(len(layers) - 2):
            layer_seqs.append(BasicBlock(layers[i], layers[i+1], bn=bn, dropout=dropout))
        layer_seqs.append(nn.Linear(layers[-2], layers[-1]))
        self.fc = nn.Sequential(
            *layer_seqs
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class AttClassifier(nn.Module):
    def __init__(self, att_dim, att_head, att_layer, n_concat, input_dim=39, output_dim=41, dropout=0.0):
        super(AttClassifier, self).__init__()
        self.n_concat = n_concat
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.transformer = torch.nn.Transformer(
            att_dim, att_head, num_encoder_layers=att_layer, num_decoder_layers=att_layer, dropout=dropout,
            batch_first=True)
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, att_dim),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Linear(self.n_concat * att_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.n_concat, self.input_dim)
        att_input = self.input_layer(x)
        att_output = self.transformer(att_input)
        att_output = att_output.reshape(-1, self.att_dim * self.n_concat)
        all_output = self.output_layer(att_output)
        return all_output

    def move_to(self, device):
        self.input_layer.to(device)
        self.output_layer.to(device)
        for layer in self.attention:
            layer.to(device)
        self.pe.to(device)


def hyper_parameters():
    parameters = {
        'concat_nframes': 21,
        'train_ratio': 0.8,
        'seed': 0,
        'batch_size': 512,
        'num_epochs': 10,
        'learning_rate': 0.0001,
        'dropout': 0.25,
        'bn': False,
        'l2': 0.0,
        'model_path': './ckpt/work2/model.ckpt',
        'data_root': './data/work2/',
        'layers': (128, 4, 2),
    }

    return parameters


def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    # 滚动平移，第一个或者最后一个元素重复，其他的前移或后移
    if n < 0:
        # 后移
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        # 前移
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


def get_test_data(params):
    feat_dir = os.path.join(params['data_root'], './libriphone/feat')
    phone_path = os.path.join(params['data_root'], './libriphone')
    test_X = preprocess_data(split='test', feat_dir=feat_dir, phone_path=phone_path,
                             concat_nframes=params['concat_nframes'])
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False)
    return test_loader


def train(train_loader, val_loader=None, params={}, device='cpu', log_dir='./'):
    model = AttClassifier(params['layers'][0], params['layers'][1], params['layers'][2], params['concat_nframes'],
                       dropout=params['dropout'],)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
    writer = SummaryWriter(os.path.join('/data/lifeinan/logs/hy/', log_dir))

    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(params['num_epochs']):
        train_acc = 0.0
        train_cnt = 0
        train_loss = 0.0
        val_acc = 0.0
        val_cnt = 0
        val_loss = 0.0

        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_cnt += len(labels)
            train_loss += loss.item()

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)

                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (val_pred.detach() == labels.detach()).sum().item()
                    val_cnt += len(labels)
                    val_loss += loss.item()
            print(f"[{epoch+1}/{params['num_epochs']} Train Acc: {train_acc/train_cnt}"
                  f"Loss: {train_loss/len(train_loader)} | Val Acc: {val_acc/val_cnt}"
                  f"Loss: {val_loss/len(val_loader)}")
            writer.add_scalar('train/acc', train_acc / train_cnt, epoch+1)
            writer.add_scalar('train/loss', train_loss / len(train_loader), epoch+1)
            writer.add_scalar('val/acc', val_acc / val_cnt, epoch+1)
            writer.add_scalar('val/loss', val_loss / len(val_loader), epoch+1)

            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss
                best_cnt = val_cnt
                torch.save(model.state_dict(), params['model_path'])
                print(f'Saving model with acc {best_acc/best_cnt}')
        else:
            print(f"[{epoch+1}/{params['num_epochs']} Train Acc: {train_acc/train_cnt}"
                  f"Loss: {train_loss/len(train_loader)}")
            writer.add_scalar('train/acc', train_acc / train_cnt, epoch+1)
            writer.add_scalar('train/loss', train_loss / len(train_loader), epoch+1)
    hparams = {
        'batch_size': params['batch_size'],
        'dropout': params['dropout'],
        'layers': ','.join(map(str, params['layers'])),
        'learning_rate': params['learning_rate'],
        'num_epochs': params['num_epochs'],
        'concat_nframes': params['concat_nframes'],
        'optimizer': 'adamw',
        'bn': params['bn'],
        'l2': params['l2'],
    }
    if val_loader is None:
        torch.save(model.state_dict(), params['model_path'])
        metrics = {
            'acc': train_acc / train_cnt,
            'loss': train_loss / len(train_loader),
        }
        writer.add_hparams(hparams, metrics)
    else:
        metrics = {
            'acc': best_acc / best_cnt,
            'loss': best_loss / best_cnt,
        }
        writer.add_hparams(hparams, metrics)


def pred(test_loader, params, device='cpu'):
    model = AttClassifier(params['layers'][0], params['layers'][1], params['layers'][2], params['concat_nframes'],
                          dropout=params['dropout'])
    model.to(device)
    model.load_state_dict(torch.load(params['model_path']))
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)
            outputs = model(features)
            _, test_pred = torch.max(outputs, 1)
            predictions.append(test_pred.cpu().numpy())
    prediction = np.concatenate(predictions)
    with open('./result/work2/prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(prediction):
            f.write(f'{i},{y}\n')


def main():
    args = get_parser().parse_args()
    if args.timestamp:
        log_dir = args.trail_id + '_' + str(int(datetime.datetime.now().timestamp()))
    else:
        log_dir = args.trail_id
    params = hyper_parameters()
    device = get_device('cuda:4', verbose=True)
    set_rand_seed(params['seed'])
    if args.mode == 'train' or args.mode == 'both':
        train_loader, val_loader = get_data(params)
        train(train_loader, val_loader, params=params, device=device, log_dir=log_dir)
    elif args.mode == 'pred' or args.mode == 'both':
        test_loader = get_test_data(params)
        pred(test_loader, params, device=device)


if __name__ == '__main__':
    main()
