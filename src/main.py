"""
Main script to train the model and create encodings
"""

import os
from torch import rand, reshape, save, load, from_numpy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from src.adl_vod_encoder.data_io.vod_data_loaders import VodTempPrecDataset
from src.adl_vod_encoder.models.autoencoders import DeepConvTempPrecAutoencoder


if __name__ == "__main__":

    temp_resolution = 'weekly'
    model_name = 'DeepConvTempPrecAutoencoder'

    in_path = '/data/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_{}.nc'.format(temp_resolution)
    in_path_tp = '/data/USERS/lmoesing/vod_encoder/data/era5mean.nc'
    model_save_path = '/data/USERS/lmoesing/vod_encoder/models/model_{}_{}.pt'.format(temp_resolution, model_name)
    output_save_path = '/data/USERS/lmoesing/vod_encoder/output/output_{}_{}.nc'.format(temp_resolution, model_name)

    ## settings
    train = True
    encoding_size = 32
    num_clusters = 30
    device = ["cpu", 'cuda:0'][1]

    try:
        os.makedirs(os.path.dirname(model_save_path))
    except FileExistsError:
        pass

    # load ds and model
    ds = VodTempPrecDataset(in_path, in_path_tp, equalyearsize=False)
    model = DeepConvTempPrecAutoencoder(ds, encoding_size, batch_size=512)

    if train:
        # train model
        model = model.to(device)
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
        trainer = pl.Trainer(max_epochs=100, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,
                             progress_bar_refresh_rate=10,
                             callbacks=[early_stop_callback]
                             )
        trainer.fit(model)
        save(model.state_dict(), model_save_path)

    model.load_state_dict(load(model_save_path))
    model.eval()
    model = model.to(device)

    encodings = model.encode_ds(ds)
    ds.add_encodings(encodings)

    predictions = model.predict_ds(ds)
    ds.add_predictions(predictions)

    loss = model.loss_all(predictions, ds)
    loss_origscale = model.loss_all(predictions, ds, origscale=True)
    ds.add_images(loss)
    ds.add_images(loss_origscale)

    cluster_ids_x = model.cluster_encodings(encodings, num_clusters)
    ds.add_image(cluster_ids_x, 'cluster_ids')
    ds.flush(output_save_path)



