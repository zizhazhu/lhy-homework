import os
import csv

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from util.util import set_rand_seed, get_device, plot_learning_curve, plot_pred


def prep_dataloader(path, mode, batch_size, n_jobs=0, selection=None):
    """ Generates a dataset, then is put into a dataloader. """
    dataset = COVID19Dataset(path, mode=mode, selection=selection)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


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


def train(tr_set, dv_set, model, config, device):
    """ DNN training """

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f}, train_loss = {:.4f})'
                  .format(epoch + 1, min_mse, loss_record['train'][-1]))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


def feature_selection(dataset):
    x = dataset.data
    y = dataset.target
    select = SelectKBest(f_regression, k=20).fit(x, y).get_support()
    return select


def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds


def hyper_parameters():
    parameters = {
        'n_epochs': 3000,                # maximum number of epochs
        'batch_size': 64,               # mini-batch size for dataloader
        'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
        'layers': (64, 64, 64),
        'dropout': None,
        'bn': True,
        'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 0.001,
            # 'momentum': 0.9,
            'weight_decay': 0.1,
        },
        'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
        'save_path': 'models/model.pth'  # your model will be saved here
    }
    return parameters


def main():
    train_data_path = './data/covid.train.csv'
    test_data_path = './data/covid.test.csv'
    set_rand_seed()
    device = get_device()                 # get the current available device ('cpu' or 'cuda')
    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/

    params = hyper_parameters()

    dataset = COVID19Dataset(train_data_path, 'train')
    select = feature_selection(dataset)

    train_dataset = prep_dataloader(train_data_path, 'train', params['batch_size'], selection=select)
    valid_dataset = prep_dataloader(train_data_path, 'dev', params['batch_size'], selection=select)
    test_dataset = prep_dataloader(test_data_path, 'test', params['batch_size'], selection=select)

    model = NeuralNet(train_dataset.dataset.dim, layers=params['layers'], dropout=params['dropout']).to(device)
    model_loss, model_loss_record = train(train_dataset, valid_dataset, model, params, device)

    plot_learning_curve(model_loss_record, title='deep model')
    del model
    model = NeuralNet(train_dataset.dataset.dim, layers=params['layers'], dropout=params['dropout']).to(device)
    ckpt = torch.load(params['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    plot_pred(valid_dataset, model, device)  # Show prediction on the validation set
    def save_pred(preds, file):
        ''' Save predictions to specified file '''
        print('Saving results to {}'.format(file))
        with open(file, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'tested_positive'])
            for i, p in enumerate(preds):
                writer.writerow([i, p])

    preds = test(test_dataset, model, device)  # predict COVID-19 cases with your model
    save_pred(preds, 'result/pred.csv')         # save prediction file to pred.csv


if __name__ == '__main__':
    main()
