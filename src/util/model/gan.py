import torch.nn as nn


class DCNNGenerator(nn.Module):
    def __init__(self, in_dim, feature_dim=64, bn=True):
        super().__init__()
        self._bn = bn

        l1_layers = [

            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
        ]
        if self._bn:
            l1_layers.append(nn.BatchNorm1d(feature_dim * 8 * 4 * 4))
        l1_layers.append(nn.ReLU(True))
        self.l1 = nn.Sequential(*l1_layers)
        # 在使用这层前会先调整view到(batch_size, feature_dim * 8, 4, 4)
        # 逐级减少feature_dim，扩大height和width
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),  # (batch, feature_dim * 16, 8, 8)
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),  # (batch, feature_dim * 16, 16, 16)
            self.dconv_bn_relu(feature_dim * 2, feature_dim),  # (batch, feature_dim * 16, 32, 32)
        )
        # change to image
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        dconv_layers = [
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),  # double height and width
        ]
        if self._bn:
            dconv_layers.append(nn.BatchNorm2d(out_dim))
        dconv_layers.append(nn.ReLU(True))
        return nn.Sequential(*dconv_layers)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y


class Discriminator(nn.Module):
    def __init__(self, in_dim, feature_dim=64, linear=True, bn=True):
        super(Discriminator, self).__init__()
        self._bn = bn
        model_seq = [
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1),       # (batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                 # (batch, 3, 16, 16)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),             # (batch, 3, 8, 8)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),             # (batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
        ]
        if not linear:
            model_seq.append(nn.Sigmoid())
        self.l1 = nn.Sequential(*model_seq)
        self.apply(weights_init)

    def conv_bn_lrelu(self, in_dim, out_dim):
        conv_layers = [
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
        ]
        if self._bn:
            conv_layers.append(nn.BatchNorm2d(out_dim))
        conv_layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*conv_layers)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
