"""
Just some basic setup to getmyself familiar with pytorch
Build after:
https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0

"""

import os
import numpy as np
import torch
from torch import rand, reshape, save, load, from_numpy
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from kmeans_pytorch import kmeans
from src.adl_vod_encoder.data_io.vod_data_loaders import VodDataset, VodTempPrecDataset
from src.adl_vod_encoder.data_io.output_writers import OutputWriter
from src.adl_vod_encoder.models.autoencoders import BaseModel, BaseConvAutoencoder, BaseTempPrecAutoencoder

if __name__ == "__main__":

    temp_resolution = 'weekly'
    model_name = 'BaseTempPrecAutoencoder'

    in_path = '/data/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_{}.nc'.format(temp_resolution)
    in_path_tp = '/data/USERS/lmoesing/vod_encoder/data/era5mean.nc'
    model_save_path = '/data/USERS/lmoesing/vod_encoder/models/model_{}_{}.pt'.format(temp_resolution, model_name)
    output_save_path = '/data/USERS/lmoesing/vod_encoder/output/output_{}_{}.nc'.format(temp_resolution, model_name)

    train = False
    device = ["cpu", 'cuda'][1]

    try:
        os.makedirs(os.path.dirname(model_save_path))
    except FileExistsError:
        pass

    ds = VodTempPrecDataset(in_path, in_path_tp)
    model = BaseTempPrecAutoencoder(ds, 16, batch_size=512)

    if train:
        model = model.to(device)
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
        trainer = pl.Trainer(max_epochs=100, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,
                             progress_bar_refresh_rate=10,
                             callbacks=[early_stop_callback]
                             )
        trainer.fit(model)
        save(model.state_dict(), model_save_path)

    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(load(model_save_path))
    model.eval()
    model = model.to(device)

    encodings = model.encode_ds(ds)
    predictions = model.predict_ds(ds)

    ds.add_predictions(predictions)
    ds.add_encodings(encodings)
    loss = model.loss_all(predictions, ds)
    loss_origscale = model.loss_all(predictions, ds, origscale=True)
    ds.add_images(loss)
    ds.add_images(loss_origscale)
    cluster_ids_x, cluster_centers = kmeans(torch.from_numpy(encodings), 10)
    ds.add_image(cluster_ids_x.detach().numpy(), 'cluster_ids')
    ds.flush(output_save_path)
