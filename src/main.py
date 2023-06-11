"""
Main script to train the model and create encodings
"""

import os
from torch import rand, reshape, save, load, from_numpy, tanh, nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from src.adl_vod_encoder.data_io.vod_data_loaders import VodTempPrecDataset, VodDataset, VODTempPrecFullTsDataset
from src.adl_vod_encoder.models.autoencoders import DeepConvTempPrecAutoencoder, \
    ShallowConvAutoencoder, BaseModel, BaseTempPrecAutoencoder, DeepConvAutoencoder, VeryDeepConvAutoencoder,\
    MonthlyShallowConvAutoencoder, MonthlyVeryDeepConvAutoencoder, MontlyLstmAutencoder, Monthly4DeepConvAutoencoder

from src.adl_vod_encoder.models.regressive_models import BaseRegressiveModel, LSTMRegressiveModel,\
    WideConvRegressiveModel, UNet, Autoregressive_UNet
from src.adl_vod_encoder.plotting.make_pretty_plots import RegressionPlotter as Plotter
from copy import deepcopy

class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(deepcopy(trainer.callback_metrics))
        print(self.metrics[-1])


if __name__ == "__main__":

    ## settings
    train = True
    create_nc = True
    create_plots = True
    encoding_size = 32
    activation_fun = nn.ReLU()
    num_clusters = 32
    noise=0.
    dropout=0.
    device = ["cpu", 'cuda:0'][1]

    temp_resolution = 'monthly'
    model_class = Autoregressive_UNet
    suffix = "1_16"
    model_name = "{}_e{}_{}_n{}_d{}_{}".format(model_class.__name__, encoding_size, "tanh", noise, dropout, suffix)

    in_path = '/data/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_{}.nc'.format(temp_resolution)
    in_path_tp = '/data/USERS/lmoesing/vod_encoder/data/era5_monthly.nc'
    model_save_path = '/data/USERS/lmoesing/vod_encoder/models/model_{}_{}.pt'.format(temp_resolution, model_name)
    trainer_save_path = '/data/USERS/lmoesing/vod_encoder/models/trainer_{}_{}.pt'.format(temp_resolution, model_name)
    output_save_path = '/data/USERS/lmoesing/vod_encoder/output/output_{}_{}.nc'.format(temp_resolution, model_name)

    if train or create_nc:
        try:
            os.makedirs(os.path.dirname(model_save_path))
        except FileExistsError:
            pass
        # load ds and model
        ds = VODTempPrecFullTsDataset(in_path, in_path_tp, equalyearsize=False)
        # ds.load_temp_data(in_path_tp)
        model = model_class(ds, batch_size=8192, activation_fun=activation_fun, noise=noise, dropout=dropout)

    if train:
        # train model
        model = model.to(device)
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=True, mode='min')
        mcb = MetricsCallback()
        trainer = pl.Trainer(max_epochs=100, min_epochs=2,
                             callbacks=[early_stop_callback, mcb]
                             )
        trainer.fit(model)
        print("Saving model ...")
        save(model.state_dict(), model_save_path)
        save(mcb, trainer_save_path)
        print("Saving model complete")

    if create_nc:
        model.load_state_dict(load(model_save_path))
        model.eval()
        device = ["cpu", 'cuda:0'][0]
        model = model.to(device)
        print("Generating output")
        # encodings = model.encode_ds(ds)
        # ds.add_encodings(encodings)

        for x in [1., 2.]:
            td = x
            pred_adj = model.predict_td_effect(ds, td=td, pf=1.0)
            ds.add_tempadjusted_predictions(pred_adj, td, 1.0)

        pred_adj = model.predict_td_effect(ds, td=0.0, pf=0.8)
        ds.add_tempadjusted_predictions(pred_adj, 0.0, 0.8)
        pred_adj = model.predict_td_effect(ds, td=0.0, pf=1.2)
        ds.add_tempadjusted_predictions(pred_adj, 0.0, 1.2)

        pred_adj = model.predict_td_effect(ds, td=0.0, pf=1.0)
        ds.add_tempadjusted_predictions(pred_adj, 0.0, 1.0)

        predictions = model.predict_ds(ds)
        ds.add_predictions(predictions)

        loss = model.loss_all(predictions, ds)
        loss_origscale = model.loss_all(predictions, ds, origscale=True)
        ds.add_images(loss)
        ds.add_images(loss_origscale)

        # cluster_ids_x = model.cluster_encodings(encodings, num_clusters)
        # ds.add_image(cluster_ids_x, 'cluster_ids')
        ds.flush(output_save_path)

    if create_plots:
        plotter = Plotter(output_save_path, trainer_save_path)
        plotter.plot_all()



