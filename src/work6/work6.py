import os

import torch.nn

from util import common_parser, get_device, set_rand_seed
import util


def get_args():
    parser = common_parser()
    return parser.parse_args()


def hyper_parameters():
    parameters = {
        'seed': 0,
        'batch_size': 64,
        'num_epochs': 10,
        'learning_rate': 0.0001,
        'l2': 0.01,
    }

    return parameters


def main():
    args = get_args()
    params = hyper_parameters()
    device = get_device('cuda:4')
    set_rand_seed(params['seed'])

    generator = util.model.Generator(100)
    discriminator = util.model.Discriminator(3)
    criterion = torch.nn.BCELoss()
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])

    train_loader = util.dataset.crypko_dataloader(args.data_dir, params['batch_size'])
    trainer = util.train.GANTrainer(generator, discriminator, criterion, g_optimizer, d_optimizer, device,
                                    model_path=args.model_dir)
    trainer.train(train_loader, params['num_epochs'], verbose=True)


if __name__ == '__main__':
    main()
