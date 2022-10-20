import os

import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm


class GANTrainer:

    def __init__(self, generator, discriminator, criterion, g_optimizer, d_optimizer,
                 device='cpu', writer=None, model_path=None):
        self._generator = generator.to(device)
        self._discriminator = discriminator.to(device)
        self._device = device
        self._criterion = criterion
        self._g_optimizer = g_optimizer
        self._d_optimizer = d_optimizer
        self._writer = writer
        self._model_path = model_path
        self._global_step = 0

        self._z_fixed = torch.autograd.Variable(torch.randn(64, 100)).to(self._device)

    def train(self, train_loader, n_epochs=1, n_critic=8, verbose=False):
        for epoch in range(n_epochs):
            for i, data in enumerate(tqdm(train_loader)):
                img = data.to(self._device)
                batch_size = img.shape[0]

                # Train D
                self._discriminator.train()
                z = torch.autograd.Variable(torch.randn(batch_size, 100)).to(self._device)
                real_image = img
                real_labels = torch.ones(batch_size).to(self._device)
                real_logits = self._discriminator(real_image)
                real_loss = self._criterion(real_logits, real_labels)

                fake_image = self._generator(z)
                fake_labels = torch.zeros(batch_size).to(self._device)
                fake_logits = self._discriminator(fake_image)
                fake_loss = self._criterion(fake_logits, fake_labels)

                d_loss = real_loss + fake_loss

                self._discriminator.zero_grad()
                d_loss.backward()
                self._d_optimizer.step()

                # Train G
                # train generator every n_critic steps
                if self._global_step % n_critic == 0:
                    self._generator.train()
                    z = torch.autograd.Variable(torch.randn(batch_size, 100)).to(self._device)
                    fake_image = self._generator(z)
                    fake_labels = torch.ones(batch_size).to(self._device)
                    fake_logits = self._discriminator(fake_image)
                    g_loss = self._criterion(fake_logits, fake_labels)

                    self._generator.zero_grad()
                    g_loss.backward()
                    self._g_optimizer.step()

                self._global_step += 1

            self._generator.eval()
            fake_images = (self._generator(self._z_fixed).data + 1) / 2
            # TODO: use tensorboard save image
            filename = os.path.join(self._model_path, f'epoch_{epoch}.png')
            print(f'Discriminator Loss: {d_loss.data:.4f}, Generator Loss: {g_loss.data:.4f}')
            torchvision.utils.save_image(fake_images, filename, nrow=8)

            if verbose:
                grid_img = torchvision.utils.make_grid(fake_images, nrow=8)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid_img.permute(1, 2, 0))
                plt.show()

