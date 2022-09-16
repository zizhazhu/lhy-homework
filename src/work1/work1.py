import os
import csv

from sklearn.feature_selection import SelectKBest, f_regression
import torch

from util.dataset.covid19 import prep_dataloader, COVID19Dataset
from util.model.nn import NeuralNet
from util.util import set_rand_seed, get_device, plot_learning_curve, plot_pred


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


def train(train_loader, val_loader, params, device='cpu'):
    n_epochs = params['n_epochs']  # Maximum number of epochs

    model = NeuralNet(train_loader.dataset.dim, layers=params['layers'], dropout=params['dropout']).to(device)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), **params['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in train_loader:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = criterion(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(val_loader, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f}, train_loss = {:.4f})'
                  .format(epoch + 1, min_mse, loss_record['train'][-1]))
            torch.save(model.state_dict(), params['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > params['early_stop']:
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


def feature_selection(train_data_path):
    dataset = COVID19Dataset(train_data_path, 'train')
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


def main():
    train_data_path = './data/covid.train.csv'
    test_data_path = './data/covid.test.csv'
    set_rand_seed()
    device = get_device()                 # get the current available device ('cpu' or 'cuda')
    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/

    params = hyper_parameters()

    select = feature_selection(train_data_path)
    train_loader = prep_dataloader(train_data_path, 'train', params['batch_size'], selection=select)
    valid_loader = prep_dataloader(train_data_path, 'dev', params['batch_size'], selection=select)
    test_loader = prep_dataloader(test_data_path, 'test', params['batch_size'], selection=select)

    model_loss, model_loss_record = train(train_loader, valid_loader, params, device)

    plot_learning_curve(model_loss_record, title='deep model')
    model = NeuralNet(train_loader.dataset.dim, layers=params['layers'], dropout=params['dropout']).to(device)
    ckpt = torch.load(params['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    plot_pred(valid_loader, model, device)  # Show prediction on the validation set
    def save_pred(preds, file):
        ''' Save predictions to specified file '''
        print('Saving results to {}'.format(file))
        with open(file, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'tested_positive'])
            for i, p in enumerate(preds):
                writer.writerow([i, p])

    preds = test(test_loader, model, device)  # predict COVID-19 cases with your model
    save_pred(preds, 'result/pred.csv')         # save prediction file to pred.csv


if __name__ == '__main__':
    main()
