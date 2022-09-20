import torch.nn as nn
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self._model = model
        self._device = device
        self._criterion = criterion
        self._optimizer = optimizer

    def train_and_eval(self, train_loader, val_loader, n_epochs=1):
        for epoch in range(n_epochs):
            self.train(train_loader)
            self.eval(val_loader)

    def train(self, train_loader, n_epochs=1):
        for epoch in range(n_epochs):
            self._model.train()
            train_loss = []
            train_acc = []

            for batch in tqdm(train_loader):
                features, labels = batch
                features = features.to(self._device)
                labels = labels.to(self._device)

                logits = self._model(features)
                loss = self._criterion(logits, labels)
                self._optimizer.zero_grad()
                loss.backward()

                # TODO: abstract it
                grad_norm = nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10)
                self._optimizer.step()

                acc = (logits.argmax(dim=-1) == labels).float().mean()
                train_loss.append(loss.item())
                train_acc.append(acc)

            train_loss_avg = sum(train_loss) / len(train_loss)
            train_acc_avg = sum(train_acc) / len(train_acc)

    def eval(self, val_loader):
        self._model.eval()
        pass

