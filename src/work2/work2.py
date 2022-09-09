import os
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trail_id', type=str, default='test')
    parser.add_argument('--timestamp', action='store_true')
    parser.add_argument('mode', type=str)
    return parser


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
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, layers, dropout=0.5):
        super(Classifier, self).__init__()
        layer_seqs = []
        for i in range(len(layers) - 2):
            layer_seqs.append(BasicBlock(layers[i], layers[i+1], dropout=dropout))
        layer_seqs.append(nn.Linear(layers[-2], layers[-1]))
        self.fc = nn.Sequential(
            *layer_seqs
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def hyper_parameters():
    parameters = {
        'concat_nframes': 21,
        'train_ratio': 0.8,
        'seed': 0,
        'batch_size': 512,
        'num_epochs': 20,
        'learning_rate': 0.0001,
        'dropout': 0.25,
        'model_path': './ckpt/work2/model.ckpt',
        'data_root': './data/work2/',
    }
    parameters['layers'] = [parameters['concat_nframes'] * 39, 512, 256, 128, 41]

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
    model = Classifier(params['layers'], dropout=params['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
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
    model = Classifier(params['layers']).to(device)
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


def same_seeds(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = get_parser().parse_args()
    if args.timestamp:
        log_dir = args.trail_id + '_' + str(int(datetime.datetime.now().timestamp()))
    else:
        log_dir = args.trail_id
    params = hyper_parameters()
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    print(f'Use device: {device}')
    same_seeds(params['seed'])
    if args.mode == 'train' or args.mode == 'both':
        train_loader, val_loader = get_data(params)
        train(train_loader, val_loader, params=params, device=device, log_dir=log_dir)
    elif args.mode == 'pred' or args.mode == 'both':
        test_loader = get_test_data(params)
        pred(test_loader, params, device=device)


if __name__ == '__main__':
    main()
