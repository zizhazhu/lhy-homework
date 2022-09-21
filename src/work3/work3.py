import torch
import torch.nn as nn

import util
from util import get_device, set_rand_seed


def main():
    device = get_device('cuda:4')
    set_rand_seed(params['seed'])
    model = model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW()
    trainer = util.train.Trainer(model, criterion, )


if __name__ == '__main__':
    main()
