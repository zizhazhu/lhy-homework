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
        'batch_size': 128,
        'num_epochs': 10,
        'learning_rate': 0.0001,
        'l2': 0.01,
        'n_critic': 1,
        'n_latent': 100,
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
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=params['learning_rate'], betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=params['learning_rate'], betas=(0.5, 0.999))

    train_loader = util.dataset.crypko_dataloader(args.data_dir, params['batch_size'])
    trainer = util.train.GANTrainer(
        generator, discriminator, criterion, g_optimizer, d_optimizer, n_latent=params['n_latent'],
        device=device, model_path=args.model_dir, output_dir=args.output_dir,
    )
    trainer.train(train_loader, params['num_epochs'], params['n_critic'], verbose=True)


if __name__ == '__main__':
    main()
