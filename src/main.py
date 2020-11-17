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
from src.adl_vod_encoder.data_io.vod_data_loaders import VodDataset
from src.adl_vod_encoder.data_io.output_writers import OutputWriter
from src.adl_vod_encoder.models.autoencoders import BaseModel, BaseConvAutoencoder

if __name__ == "__main__":

    temp_resolution = 'weekly'
    model_name = 'BaseConvAutoencoder'

    in_path = '/data/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_{}.nc'.format(temp_resolution)
    model_save_path = '/data/USERS/lmoesing/vod_encoder/models/model_{}_{}.pt'.format(temp_resolution, model_name)
    encodings_save_path = '/data/USERS/lmoesing/vod_encoder/output/output_{}_{}.nc'.format(temp_resolution, model_name)

    mode = 'load'
    cluster = True
    try:
        os.makedirs(os.path.dirname(model_save_path))
    except FileExistsError:
        pass
    ds = VodDataset(in_path)

    device = ["cpu", 'cuda'][1]

    model = BaseConvAutoencoder(ds, 4, batch_size=512).to(device)

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

    device = 'cpu'
    prediction_list = []
    model = model.to(device)
    for i, batch in enumerate(DataLoader(ds, batch_size=model.batch_size, num_workers=1)):
        output = model(batch[0])
        prediction_list.append(output.detach().numpy())

    encodings = np.concatenate(prediction_list)

    writer = OutputWriter(ds)
    writer.add_encodings(encodings)
    cluster_ids_x, cluster_centers = kmeans(torch.from_numpy(encodings), 10)
    writer.add_clusteridx(cluster_ids_x.detach().numpy())
    writer.flush(encodings_save_path)
