"""
Just some basic setup to getmyself familiar with pytorch
Build after:
https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0

"""

import os
import xarray as xr
import numpy as np
from torch import nn, optim, rand, reshape, save
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.callbacks import EarlyStopping


class VodDataset(Dataset):
    """
    VOD dataset. Current limitations:
        - only post 1987 as it is incomplete
        - only ts without any nan value
    """
    def __init__(self, in_path):
        da = xr.open_dataarray(in_path)
        da = da[da['time.year'] > 1987]
        self.data = da.values[:, ~da.isnull().any('time')].T.astype(np.float32)
        self.time = da['time']
        self.sample_dim = self.data.shape[1]

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.data.shape[0]


class OurModel(pl.LightningModule):
    """
    Minimalistic autoencoder. Only the encoding layer, nothing else. Cant handle missing values.
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7):
        super(OurModel, self).__init__()
        self.linear = nn.Linear(dataset.sample_dim, encoding_dim)
        self.linear2 = nn.Linear(encoding_dim, dataset.sample_dim)
        self.lr = 0.001
        self.batch_size = 512
        trainsize = int(len(ds)*train_size_frac)
        valsize = len(ds) - trainsize
        self.dataset_train, self.dataset_val = random_split(dataset, [trainsize, valsize])

    def forward(self, x):
        encoding = self.linear(x)
        output = self.linear2(encoding)
        return output

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
        return loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def val_dataloader(self):
        loader = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)
        return loader

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        return {'val_loss': loss, 'log': {'val_loss': loss}}

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_acc in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results

    
if __name__ == "__main__":
    in_path = '/data-write/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_monthly.nc'
    model_save_path = '/data-write/USERS/lmoesing/vod_encoder/models/model0.pt'
    try:
        os.makedirs(os.path.dirname(model_save_path))
    except FileExistsError:
        pass
    ds = VodDataset(in_path)

    device = 'cpu'
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
    model = OurModel(ds, 4).to(device)
    trainer = pl.Trainer(max_epochs=100, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,
                         progress_bar_refresh_rate=10,
                         callbacks=[early_stop_callback]
                         )
    trainer.fit(model)
    save(model.state_dict(), model_save_path)

