import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import util
from util import get_device, set_rand_seed, common_parser


def get_args():
    parser = common_parser()
    return parser.parse_args()


def hyper_parameters():
    parameters = {
        'seed': 0,
        'batch_size': 512,
        'num_epochs': 10,
        'learning_rate': 0.0001,
        'dropout': 0.25,
        'bn': False,
        'l2': 0.0,
        'layers': (128, 4, 2),
    }

    return parameters


def main():
    args = get_args()
    params = hyper_parameters()

    device = get_device('cuda:4')
    set_rand_seed(params['seed'])

    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    model_path = os.path.join(args.model_dir, 'model.ckpt')

    model = util.model.CNNClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
    trainer = util.train.Trainer(model, criterion, optimizer, device, model_path=model_path)

    if args.mode == 'train':
        train_dataset = util.dataset.FoodDataset(os.path.join(args.data_dir, 'training'), train_tfm)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        valid_dataset = util.dataset.FoodDataset(os.path.join(args.data_dir, 'validation'), test_tfm)
        valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=True)
        trainer.train_and_eval(train_loader, valid_loader, params['num_epochs'], verbose=True)
    else:
        test_dataset = util.dataset.FoodDataset(os.path.join(args.data_dir, 'test'), test_tfm)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)
        model.load_state_dict(torch.load(model_path))
        predictions = trainer.predict(test_loader)


if __name__ == '__main__':
    main()
