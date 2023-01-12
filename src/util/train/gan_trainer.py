import os
import abc

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class GANTrainer:

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 n_latent=100, device='cpu', writer=None, model_path=None, output_dir=None,
                 ):
        self._generator = generator.to(device)
        self._discriminator = discriminator.to(device)
        self._device = device
        self._g_optimizer = g_optimizer
        self._d_optimizer = d_optimizer
        self._writer = writer
        self._model_path = model_path
        self._output_dir = output_dir
        self._n_latent = n_latent

        self._global_step = 0
        self._z_fixed = torch.autograd.Variable(torch.randn(64, self._n_latent)).to(self._device)

        self._writer = SummaryWriter(log_dir=self._model_path)

    @abc.abstractmethod
    def _d_loss(self, real_logits, fake_logits, real_images, fake_images):
        raise NotImplementedError()

    @abc.abstractmethod
    def _g_loss(self, fake_logits):
        raise NotImplementedError()

    @abc.abstractmethod
    def _clip_weights(self):
        pass

    def train(self, train_loader, n_epochs=1, n_critic=8, interval=100):
        for epoch in range(n_epochs):
            d_loss = g_loss = 0.0
            for i, data in enumerate(tqdm(train_loader)):
                img = data.to(self._device)
                batch_size = img.shape[0]

                # Train D
                self._discriminator.train()
                self._discriminator.zero_grad()
                z = torch.autograd.Variable(torch.randn(batch_size, self._n_latent)).to(self._device)
                real_image = img
                real_logits = self._discriminator(real_image)

                fake_image = self._generator(z)
                fake_logits = self._discriminator(fake_image)

                d_loss = self._d_loss(real_logits, fake_logits, real_image, fake_image)

                d_loss.backward()
                self._d_optimizer.step()

                self._clip_weights()

                # Train G
                # train generator every n_critic steps
                if self._global_step % n_critic == 0:
                    self._generator.train()
                    z = torch.autograd.Variable(torch.randn(batch_size, self._n_latent)).to(self._device)
                    fake_image = self._generator(z)
                    fake_logits = self._discriminator(fake_image)
                    g_loss = self._g_loss(fake_logits)

                    self._generator.zero_grad()
                    g_loss.backward()
                    self._g_optimizer.step()

                if self._global_step % interval == 0:
                    self._generator.eval()
                    fake_images = (self._generator(self._z_fixed).data + 1) / 2
                    self._writer.add_scalar('d_loss', d_loss.item(), self._global_step)
                    self._writer.add_scalar('g_loss', g_loss.item(), self._global_step)
                    self._writer.add_image('fake_image', torchvision.utils.make_grid(fake_images), self._global_step)

                self._global_step += 1

            print(f'Discriminator Loss: {d_loss.data:.4f}, Generator Loss: {g_loss.data:.4f}')

            if self._model_path:
                torch.save(self._generator.state_dict(), os.path.join(self._model_path, f'generator_{epoch}.pth'))
                torch.save(self._discriminator.state_dict(), os.path.join(self._model_path,
                                                                          f'discriminator_{epoch}.pth'))


class VanillaGANTrainer(GANTrainer):

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, criterion,
                 n_latent=100, device='cpu', writer=None, model_path=None, output_dir=None,
                 ):
        super().__init__(generator, discriminator, g_optimizer, d_optimizer,
                         n_latent, device, writer, model_path, output_dir)
        self._criterion = criterion

    def _d_loss(self, real_logits, fake_logits, real_images, fake_images):
        real_labels = torch.ones(real_logits).to(self._device)
        real_loss = self._criterion(real_logits, real_labels)
        fake_labels = torch.zeros(fake_logits).to(self._device)
        fake_loss = self._criterion(fake_logits, fake_labels)
        return real_loss + fake_loss

    def _g_loss(self, fake_logits):
        fake_labels = torch.ones(fake_logits).to(self._device)
        return self._criterion(fake_logits, fake_labels)


class WGANTrainer(GANTrainer):

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 n_latent=100, device='cpu', writer=None, model_path=None, output_dir=None, clip_value=0.01):
        super().__init__(generator, discriminator, g_optimizer, d_optimizer,
                         n_latent, device, writer, model_path, output_dir)
        self._clip_value = clip_value

    def _d_loss(self, real_logits, fake_logits, real_images, fake_images):
        return -torch.mean(real_logits) + torch.mean(fake_logits)

    def _g_loss(self, fake_logits):
        return -torch.mean(fake_logits)

    def _clip_weights(self):
        for p in self._discriminator.parameters():
            p.data.clamp_(-self._clip_value, self._clip_value)


class GPGANTrainer(GANTrainer):

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 n_latent=100, device='cpu', writer=None, model_path=None, output_dir=None, gp_weight=10,
                 ):
        super().__init__(generator, discriminator, g_optimizer, d_optimizer,
                         n_latent, device, writer, model_path, output_dir)
        self._gp_weight = gp_weight

    def _gradient_penalty(self, real, fake):
        batch_size = real.shape[0]
        # choose random interpolation point
        alpha = torch.rand(batch_size, 1, 1, 1).to(self._device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        # calculate probability of interpolates
        interpolates_prob = self._discriminator(interpolates)
        # calculate gradients of probabilities with respect to interpolates
        fake = torch.autograd.Variable(torch.ones(interpolates_prob.size()), requires_grad=True).to(self._device)
        gradients = torch.autograd.grad(outputs=interpolates_prob, inputs=interpolates,
                                        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _d_loss(self, real_logits, fake_logits, real_images, fake_images):
        gradient_penalty = self._gp_weight * self._gradient_penalty(real_images, fake_images)
        return -torch.mean(real_logits) + torch.mean(fake_logits) + gradient_penalty

    def _g_loss(self, fake_logits):
        return -torch.mean(fake_logits)


