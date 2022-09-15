import os
import datetime
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from util import get_device, set_rand_seed
from util.dataset.libri import prep_dataloader
from util.model.attention import AttClassifier


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trail_id', type=str, default='test')
    parser.add_argument('--timestamp', action='store_true')
    parser.add_argument('mode', type=str)
    return parser


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


def train(train_loader, val_loader=None, params={}, device='cpu', log_dir='./'):
    model = AttClassifier(params['layers'][0], params['layers'][1], params['layers'][2], params['concat_nframes'],
                          dropout=params['dropout'], )
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

    feat_dir = os.path.join(params['data_root'], './libriphone/feat')
    phone_path = os.path.join(params['data_root'], './libriphone')
    if args.mode == 'train' or args.mode == 'both':
        train_loader = prep_dataloader(feat_dir, phone_path, params['concat_nframe'], params['train_ratio'],
                                       batch_size=params['batch_size'], mode='train')
        val_loader = prep_dataloader(feat_dir, phone_path, params['concat_nframe'], params['train_ratio'],
                                     batch_size=params['batch_size'], mode='val')
        train(train_loader, val_loader, params=params, device=device, log_dir=log_dir)
    elif args.mode == 'pred' or args.mode == 'both':
        test_loader = prep_dataloader(feat_dir, phone_path, params['concat_nframe'],
                                      batch_size=params['batch_size'], mode='test')
        pred(test_loader, params, device=device)


if __name__ == '__main__':
    main()
