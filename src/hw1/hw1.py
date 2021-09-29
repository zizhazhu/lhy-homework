import os
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from util.util import set_rand_seed, get_device, plot_learning_curve, plot_pred


class COVID19Dataset(Dataset):
    """ Dataset for loading and preprocessing the COVID19 dataset """
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            # remove header and id
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            pass

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
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

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


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    """ Generates a dataset, then is put into a dataloader. """
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim, layers=(128, 64, 32), activation=nn.ReLU):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        seqs = []
        last_layer = input_dim
        for layer in layers:
            seqs.append(nn.Linear(last_layer, layer))
            last_layer = layer
            if activation is not None:
                seqs.append(activation())
        seqs.append(nn.Linear(last_layer, 1))
        self.net = nn.Sequential(*seqs)

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
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
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))
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


def main():
    train_data_path = './data/covid.train.csv'
    test_data_path = './data/covid.test.csv'
    set_rand_seed()
    device = get_device()                 # get the current available device ('cpu' or 'cuda')
    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
    target_only = False                   # TODO: Using 40 states & 2 tested_positive features

    # TODO: How to tune these hyper-parameters to improve your model's performance?
    config = {
        'n_epochs': 3000,                # maximum number of epochs
        'batch_size': 270,               # mini-batch size for dataloader
        'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
        'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.01,
        },
        'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
        'save_path': 'models/model.pth'  # your model will be saved here
    }
    train_dataset = prep_dataloader(train_data_path, 'train', config['batch_size'], target_only=target_only)
    valid_dataset = prep_dataloader(train_data_path, 'dev', config['batch_size'], target_only=target_only)
    test_dataset = prep_dataloader(test_data_path, 'test', config['batch_size'], target_only=target_only)

    model = NeuralNet(train_dataset.dataset.dim).to(device)
    model_loss, model_loss_record = train(train_dataset, valid_dataset, model, config, device)

    plot_learning_curve(model_loss_record, title='deep model')
    del model
    model = NeuralNet(train_dataset.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
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