import torch
from torch import nn, optim, rand, reshape, from_numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
import numpy as np
from src.adl_vod_encoder.models.layers import Split, Squeeze, Reshape

class BaseModel(pl.LightningModule):
    """
    Minimalistic autoencoder to be used for inheritance for more complex models
    Handles missing values by setting them to 0, and masking them out for loss calculation.
    Therefore it will still try to predict missing values, but wont be penalized for them.

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

    def encode_ds(self, ds):
        """
        Given a dataset, encode it
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        batch_encoding_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_encoding = self(batch[0])
            batch_encoding_list.append(batch_encoding.detach().numpy())
        encodings = np.concatenate(batch_encoding_list)
        return encodings

    def predict_ds(self, ds):
        batch_ts_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_encoding = self(batch[0])
            batch_ts_hat = self.decoder(batch_encoding)
            batch_ts_hat_list.append(batch_ts_hat.detach().numpy())
        ts_hats = np.concatenate(batch_ts_hat_list)
        return ts_hats

    def loss_all(self, predictions, ds, origscale=False):

        if origscale:
            reconstruction_loss = np.nanmean((ds.data * ds.vod_std - predictions[0] * ds.vod_std)**2,1)
            loss = {'reconstruction_loss_origscale': reconstruction_loss}
        else:

            reconstruction_loss = np.nanmean((ds.data - predictions[0])**2,1)
            loss = {'reconstruction_loss': reconstruction_loss}
        return loss


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


class BaseTempPrecAutoencoder(BaseModel):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):

        super(BaseTempPrecAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)

        self.temppredictor = nn.Linear(encoding_dim, 1)
        self.precpredictor = nn.Linear(encoding_dim, 1)

    def training_step(self, batch, batch_nb):
        x, y1, y2 = batch
        x = x.to(self.device)
        y1 = y1.to(self.device)
        y2 = y2.to(self.device)
        inputnans = x != x
        encoding = self(x)

        x_hat = self.decoder(encoding)
        t_hat = self.temppredictor(encoding)
        p_hat = self.precpredictor(encoding)

        reconstruction_loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        t_loss = F.mse_loss(t_hat.squeeze(), y1)
        p_loss = F.mse_loss(p_hat.squeeze(), y2)
        loss = reconstruction_loss + t_loss + p_loss
        return loss

    def validation_step(self, batch, batch_nb):
        x, y1, y2 = batch
        x = x.to(self.device)
        y1 = y1.to(self.device)
        y2 = y2.to(self.device)
        inputnans = x != x
        encoding = self(x)

        x_hat = self.decoder(encoding)
        t_hat = self.temppredictor(encoding)
        p_hat = self.precpredictor(encoding)

        reconstruction_loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        t_loss = F.mse_loss(t_hat.squeeze(), y1)
        p_loss = F.mse_loss(p_hat.squeeze(), y2)
        loss = reconstruction_loss + t_loss + p_loss
        self.log('val_loss', loss)

    def predict_ds(self, ds):
        batch_ts_hat_list = []
        batch_t_hat_list = []
        batch_p_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):

            batch_encoding = self(batch[0])
            batch_x_hat = self.decoder(batch_encoding)
            batch_t_hat = self.temppredictor(batch_encoding)
            batch_p_hat = self.precpredictor(batch_encoding)

            batch_ts_hat_list.append(batch_x_hat.detach().numpy())
            batch_t_hat_list.append(batch_t_hat.detach().numpy())
            batch_p_hat_list.append(batch_p_hat.detach().numpy())
        x_hats = np.concatenate(batch_ts_hat_list)
        t_hats = np.concatenate(batch_t_hat_list)
        p_hats = np.concatenate(batch_p_hat_list)
        return x_hats, t_hats, p_hats

    def loss_all(self, predictions, ds, origscale=False):

        if origscale:
            reconstruction_loss = np.nanmean((ds.data * ds.vod_std - predictions[0] * ds.vod_std)**2,1)
            t_loss = (ds.tempdata * ds.temp_std - predictions[1].squeeze() * ds.temp_std)**2
            p_loss = (ds.precdata * ds.prec_std - predictions[2].squeeze() * ds.prec_std)**2
            loss = {'reconstruction_loss_origscale': reconstruction_loss,
                    't_loss_origscale': t_loss,
                    'p_loss_origscale': p_loss}
        else:

            reconstruction_loss = np.nanmean((ds.data - predictions[0])**2,1)
            t_loss = (ds.tempdata - predictions[1].squeeze())**2
            p_loss = (ds.precdata - predictions[2].squeeze())**2
            loss = {'reconstruction_loss': reconstruction_loss,
                    't_loss': t_loss,
                    'p_loss': p_loss}
        return loss


class ConvTempPrecAutoencoder(BaseTempPrecAutoencoder):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(ConvTempPrecAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
        auxpreddim = 8

        self.temppredictor = nn.Sequential(nn.Linear(encoding_dim, auxpreddim),
                                           nn.Linear(auxpreddim, 1))
        self.precpredictor = nn.Sequential(nn.Linear(encoding_dim, auxpreddim),
                                           nn.Linear(auxpreddim, 1))

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
