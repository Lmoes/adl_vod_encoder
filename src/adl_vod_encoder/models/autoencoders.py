import torch
from torch import nn, optim, rand, reshape, from_numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split


def calc_rsquared(a, b):
    vx = a - torch.mean(a)
    vy = b - torch.mean(b)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


class Reshape(pl.LightningModule):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class Squeeze(pl.LightningModule):

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class BaseModel(pl.LightningModule):
    """
    Minimalistic autoencoder to be used for inheritance for more complex models
    Handles missing values by setting them to 0, and masking them out for loss calculation

    Base code structure from these two resources:
    https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0
    https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_autoencoder.py

    """
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        trainsize = int(len(dataset)*train_size_frac)
        valsize = len(dataset) - trainsize
        self.dataset_train, self.dataset_val = random_split(dataset, [trainsize, valsize])

        self.encoder = nn.Sequential(
            nn.Linear(dataset.sample_dim, encoding_dim),
            Squeeze()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, dataset.sample_dim),
        )

    def forward(self, x):
        # x = x.to(self.device)
        x[x != x] = 0.
        encoding = self.encoder(x[:, None])
        return encoding

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.to(self.device)
        inputnans = x != x
        encoding = self(x)

        x_hat = self.decoder(encoding)
        loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x.to(self.device)
        inputnans = x != x
        encoding = self(x)

        x_hat = self.decoder(encoding)
        loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        self.log('val_loss', loss)

    def train_dataloader(self):
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=64)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=64)
        return loader


class BaseConvAutoencoder(BaseModel):
    """
    Minimalistic convolutional autoencoder. Simplified version of https://arxiv.org/abs/2002.03624v1
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(BaseConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)

        conv1_width = 7
        conv1_nfeatures = 32
        self.encoder = nn.Sequential(
            nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=1, padding_mode='circular'),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.Flatten(),
            nn.Linear((dataset.sample_dim - conv1_width + 1) * conv1_nfeatures, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, (dataset.sample_dim - conv1_width + 1) * conv1_nfeatures),
            Reshape((-1, conv1_nfeatures, (dataset.sample_dim - conv1_width + 1))),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width),
            Squeeze()
        )