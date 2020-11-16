"""
Just some basic setup to getmyself familiar with pytorch
Build after:
https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0

"""

import os
import xarray as xr
import numpy as np
import torch
from torch import nn, optim, rand, reshape, save, load, from_numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.callbacks import EarlyStopping
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans


class VodDataset(Dataset):
    """
    VOD dataset. Current limitations:
        - only post 1987 as it is incomplete
    """
    def __init__(self, in_path, nonans=False):
        self.da = xr.open_dataarray(in_path)
        da = self.da[self.da['time.year'] > 1987]
        if nonans:
            self.tslocs = ~da.isnull().any('time')
        else:
            self.tslocs = ~da.isnull().all('time')
        self.data = da.values[:, self.tslocs].T.astype(np.float32)
        self.time = da['time']
        self.sample_dim = self.data.shape[1]
        self.out_da_list = []

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def add_encodings(self, encodings):

        encoding_dim = encodings.shape[1]
        coords = {'latent_variable': np.arange(encoding_dim), **{c: self.da.coords[c] for c in ['lat', 'lon']}}
        da = xr.DataArray(np.nan, coords, ['latent_variable', 'lat', 'lon'], 'encoding')
        da.values[:, self.tslocs] = encodings.T
        self.out_da_list.append(da)

    def add_clusteridx(self, cluster_idx):
        coords = {c: self.da.coords[c] for c in ['lat', 'lon']}
        da = xr.DataArray(np.nan, coords, ['lat', 'lon'], 'cluster_idx')
        da.values[self.tslocs] = cluster_idx
        self.out_da_list.append(da)

    def flush_outputs(self, fname):
        try:
            os.makedirs(os.path.dirname(fname))
        except FileExistsError:
            pass
        ds = xr.merge(self.out_da_list)
        ds.to_netcdf(fname)


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

class OurModel(pl.LightningModule):
    """
    Minimalistic autoencoder. Only the encoding layer, nothing else. Cant handle missing values.
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7):
        super(OurModel, self).__init__()

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

            # nn.BatchNorm1d(conv1_nfeatures)
        )
        self.lr = 0.001
        self.batch_size = 512
        trainsize = int(len(ds)*train_size_frac)
        valsize = len(ds) - trainsize
        self.dataset_train, self.dataset_val = random_split(dataset, [trainsize, valsize])

    def forward(self, x):
        x[x != x] = 0.
        encoding = self.encoder(x[:, None])
        # a = nn.Linear(4, 32*(384-7+1))(encoding)
        # b = torch.reshape(a, (512, 32, (384-7+1)))
        # c = nn.ConvTranspose1d(32, 1, 7)(b)
        # d = torch.squeeze(c)
        return encoding

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
        return loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        inputnans = x != x
        x = x.view(x.size(0), -1)
        encoding = self(x)

        x_hat = self.decoder(encoding)
        loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        return loss

    def val_dataloader(self):
        loader = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)
        return loader

    def validation_step(self, batch, batch_nb):
        x, y = batch
        inputnans = x != x
        x = x.view(x.size(0), -1)
        encoding = self(x)
        x_hat = self.decoder(encoding)
        loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        self.log('val_loss', loss)

    
if __name__ == "__main__":
    in_path = '/data-write/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_weekly.nc'
    model_save_path = '/data-write/USERS/lmoesing/vod_encoder/models/model0.pt'
    encodings_save_path = '/data-write/USERS/lmoesing/vod_encoder/encodings/encoding0.nc'
    mode = 'train'
    cluster = True
    try:
        os.makedirs(os.path.dirname(model_save_path))
    except FileExistsError:
        pass
    ds = VodDataset(in_path)

    device = 'cpu'
    model = OurModel(ds, 4).to(device)

    if mode == 'load':
        model.load_state_dict(load(model_save_path))
        model.eval()
    elif mode == 'train':
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
        trainer = pl.Trainer(max_epochs=100, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,
                             progress_bar_refresh_rate=10,
                             callbacks=[early_stop_callback]
                             )
        trainer.fit(model)
        save(model.state_dict(), model_save_path)

    encodings = model(torch.from_numpy(ds.data))

    ds.add_encodings(encodings.detach().numpy())

    cluster_ids_x, cluster_centers = kmeans(encodings, 10)
    ds.add_clusteridx(cluster_ids_x.detach().numpy())
    ds.flush_outputs(encodings_save_path)
