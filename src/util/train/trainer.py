import torch
import torch.nn as nn
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device='cpu', writer=None, model_path=None):
        self._model = model
        self._device = device
        self._criterion = criterion
        self._optimizer = optimizer
        self._writer = writer
        self._model_path = model_path

    def train_and_eval(self, train_loader, val_loader, n_epochs=1, verbose=False):
        best_acc = 0.0
        for epoch in range(n_epochs):
            train_acc_avg, train_loss_avg = self.train(train_loader)
            if verbose:
                print(f"[{epoch+1}/{n_epochs} Train Acc: {train_acc_avg} Loss: {train_loss_avg}")
            valid_acc_avg, valid_loss_avg = self.eval(val_loader)
            if verbose:
                print(f"Valid Acc: {valid_acc_avg} Loss: {valid_loss_avg}")
            if valid_acc_avg > best_acc:
                torch.save(self._model.state_dict(), self._model_path)
                best_acc = valid_acc_avg

    def train(self, train_loader, n_epochs=1, verbose=False):
        train_acc_avg = 0.0
        train_loss_avg = 0.0
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

            if verbose:
                print(f"[{epoch+1}/{n_epochs} Train Acc: {train_acc_avg} Loss: {train_loss_avg}")
            if self._writer:
                self._writer.add_scalar('acc/train', train_acc_avg, epoch+1)
                self._writer.add_scalar('loss/train', train_loss_avg, epoch+1)
        return train_acc_avg, train_loss_avg

    def eval(self, val_loader, n_epochs=1, verbose=False):
        valid_loss_avg = 0.0
        valid_acc_avg = 0.0
        for epoch in range(n_epochs):
            self._model.eval()
            valid_loss = []
            valid_acc = []

            for batch in tqdm(val_loader):
                features, labels = batch
                features = features.to(self._device)
                labels = labels.to(self._device)

                with torch.no_grad():
                    logits = self._model(features)
                loss = self._criterion(logits, labels)

                acc = (logits.argmax(dim=-1) == labels).float().mean()
                valid_loss.append(loss.item())
                valid_acc.append(acc)

            valid_loss_avg = sum(valid_loss) / len(valid_loss)
            valid_acc_avg = sum(valid_acc) / len(valid_acc)

            if verbose:
                print(f"[{epoch+1}/{n_epochs} Valid Acc: {valid_acc_avg} Loss: {valid_loss_avg}")
            if self._writer:
                self._writer.add_scalar('acc/valid', valid_acc_avg, epoch+1)
                self._writer.add_scalar('loss/valid', valid_loss_avg, epoch+1)
        return valid_acc_avg, valid_loss_avg
