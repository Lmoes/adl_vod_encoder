"""
Main script to train the model and create encodings
"""

import os
from torch import save, load
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
# set the dataset loader here
from src.adl_vod_encoder.data_io.vod_data_loaders import SMDataSet as voddataset
# set the deep learning model here
from src.adl_vod_encoder.models.autoencoders import NeighbourLSTMGapFiller as modelclass
import xarray as xr

if __name__ == "__main__":


    losssubset = "onlygaps"
    splitglobal = True
    model_name = '{}_{}_{}_splitglobal_{}'.format(modelclass.__name__, losssubset, splitglobal, "5_neigh_sm")

    # in_path = '/data/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_{}.nc'.format(temp_resolution)
    in_path = '/data/USERS/lmoesing/vod_encoder/data/STACK_C3S-SOILMOISTURE_v202012_COMBINED_MONTHLY.nc'
    in_path_tp = '/data/USERS/lmoesing/vod_encoder/data/era5mean.nc'
    model_save_path = '/data/USERS/lmoesing/vod_encoder/models/model_{}.pt'.format(model_name)
    output_save_path = '/data/USERS/lmoesing/vod_encoder/output/output_{}.nc'.format(model_name)

    ## settings
    # Whether to train a model, or skip training load an already trained one
    train = True
    # Whether to tain on CPU or GPU. Training goes usually a lot faster on the GPU, but debugging is easier on the CPU
    device = ["cpu", 'cuda:0'][1]
    # How many workers to use for data loading. Because we can anyway load all data into memory we dont need multiple, and greater 0 makes debugging messy
    num_workers = 0
    # size of batch. Usually set to as high as possible until your GPU runs out of memory
    batch_size = 1024
    # neighbours is an integer indicating how many neighbouring ts should be used
    #     0: Just center ts (use LSTMGapFiller)
    #     1: 3-neighbourhood = 9 ts total (use NeighbourLSTMGapFiller)
    #     2: 5-neighbourhood = 25 total (use NeighbourLSTMGapFiller)
    #     3: 7-neibourhood = 49 total (use NeighbourLSTMGapFiller)
    #     4: etc....
    neighbours=5
    # what to fill nans with during training. Im still unsure what the best approach is, but this worked best
    nan_fillvalue= 0.


    try:
        os.makedirs(os.path.dirname(model_save_path))
    except FileExistsError:
        pass

    # load data
    ds = voddataset(in_path, split="train",neighbours=neighbours)
    # specify model
    model = modelclass(ds, batch_size=batch_size, lr=0.001, losssubset=losssubset, splitglobal=splitglobal,
                       num_workers=num_workers, nan_fillvalue=nan_fillvalue)

    if train:
        # train model
        model = model.to(device)
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=True, mode="min")
        trainer = pl.Trainer(max_epochs=200, min_epochs=5, auto_lr_find=False, auto_scale_batch_size=False,
                             progress_bar_refresh_rate=10,
                             callbacks=[early_stop_callback]
                             )
        trainer.fit(model)
        save(model.state_dict(), model_save_path)

    ### create various outputs
    ### linear predictions
    da = xr.DataArray(ds.data)
    linear_interp = da.interpolate_na("dim_1").values
    ds.changefilter("test")

    loss_linear = model.loss_all(linear_interp, ds)
    # loss_origscale_linear = model.loss_all(linear_interp, ds, origscale=True)
    ds.add_image(loss_linear["reconstruction_loss"], "reconstruction_loss_linear")
    # ds.add_image(loss_origscale_linear["reconstruction_loss_origscale"], "reconstruction_loss_origscale_linear")
    loss_per_gaplength_linear = model.loss_per_gaplength(linear_interp, ds, "linear")
    ds.out_da_list.append(loss_per_gaplength_linear)

    ### train

    # model = modelclass(ds, batch_size=int(batch_size/4))
    ds.changefilter("train")

    model.load_state_dict(load(model_save_path))
    model.eval()
    model = model.to(device)

    predictions = model.predict_ds(ds)
    ds.add_predictions(predictions)
    loss = model.loss_all(predictions, ds)
    # loss_origscale = model.loss_all(predictions, ds, origscale=True)
    ds.add_images(loss)
    # ds.add_images(loss_origscale)


    ### test
    ds.changefilter("test")
    loss = model.loss_all(predictions, ds)
    # loss_origscale = model.loss_all(predictions, ds, origscale=True)
    ds.add_images(loss)
    # ds.add_images(loss_origscale)
    loss_per_gaplength = model.loss_per_gaplength(predictions, ds, model_name)
    ds.out_da_list.append(loss_per_gaplength)



    


    ds.flush(output_save_path)



