import torch
from torch import nn, optim, rand, reshape, from_numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
import numpy as np
from src.adl_vod_encoder.models.layers import Squeeze, Reshape, View
from src.adl_vod_encoder.models.validation_metrics import normalized_scatter_ratio
from kmeans_pytorch import kmeans
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from src.adl_vod_encoder.foundation.orthogonalization import qr_tempadjust, samedirection, adj_r2
from .autoencoders import BaseModel
import math
from collections import OrderedDict


class BaseRegressiveModel(pl.LightningModule):

    def __init__(self, dataset, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None,
                 noise=0., dropout=0.):
        super(BaseRegressiveModel, self).__init__()
        self.lr = lr
        self.noise = noise
        self.dropout = dropout
        self.batch_size = batch_size
        if activation_fun is None:
            activation_fun = F.elu
        self.activation_fun = activation_fun

        trainsize = int(len(dataset)*train_size_frac)
        valsize = len(dataset) - trainsize
        self.dataset_train, self.dataset_val = random_split(dataset, [trainsize, valsize])
        self._create_layers()

    def _create_layers(self):
        self.conv1 = nn.Conv1d(2, 1, 1, stride=1, padding_mode='circular')
        self.linear1 = nn.Linear(self.dataset_train.dataset.sample_dim, self.dataset_train.dataset.sample_dim)

    def train_dataloader(self):
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=1)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=1)
        return loader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, h, temp, prec):
        temp[temp != temp] = 0.
        prec[prec != prec] = 0.
        tp = torch.stack([temp, prec]).transpose(0,1)
        v = self.conv1(tp).squeeze()
        v = self.activation_fun(v)
        v = self.linear1(v)
        return v

    def training_step(self, batch, batch_nb):
        vod, temp, prec = batch
        vod = vod.to(self.device)
        temp = temp.to(self.device)
        prec = prec.to(self.device)

        inputnans = vod != vod

        vod_hat = self(vod, temp, prec)

        loss = F.mse_loss(vod_hat[~inputnans], vod[~inputnans])
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.training_step(batch, batch_nb)
        self.log('val_loss', loss)


    def predict_ds(self, ds):
        """
        Given a dataset, return its predicted VOD
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        batch_ts_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_ts_hat = self(*batch)
            batch_ts_hat_list.append(batch_ts_hat.cpu().detach().numpy())
        ts_hats = np.concatenate(batch_ts_hat_list)
        return ts_hats

    def loss_all(self, predictions, ds, origscale=False):
        """
        Given a dataset, retrurn the loss of it
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        if origscale:
            reconstruction_loss = np.nanmean((ds.data * ds.vod_std - predictions[0] * ds.vod_std) ** 2, 1)
            loss = {'reconstruction_loss_origscale': reconstruction_loss}
        else:

            reconstruction_loss = np.nanmean((ds.data - predictions[0]) ** 2, 1)
            loss = {'reconstruction_loss': reconstruction_loss}
        return loss

    def predict_td_effect(self, ds, td=0., pf=1.0):
        batch_ts_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch[1] = batch[1] + td / ds.temp_std
            batch[2] = (((batch[2] * ds.prec_std + ds.prec_mean) * pf) - ds.prec_mean) / ds.prec_std
            batch_ts_hat = self(*batch)
            batch_ts_hat_list.append(batch_ts_hat.cpu().detach().numpy())
        ts_hats = np.concatenate(batch_ts_hat_list)
        return ts_hats

class ConvRegressiveModel(BaseRegressiveModel):

    def _create_layers(self):
        conv1_nfeatures = 8
        self.conv1 = nn.Conv1d(2, conv1_nfeatures, 7, stride=1, padding_mode='circular', padding=math.floor(7/2))
        self.batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)

        self.conv2 = nn.Conv1d(8, 1, 1, stride=1, padding_mode='circular', padding=0)
        self.linear1 = nn.Linear(self.dataset_train.dataset.sample_dim, self.dataset_train.dataset.sample_dim)

    def forward(self, h, temp, prec):
        temp[temp != temp] = 0.
        prec[prec != prec] = 0.
        tp = torch.stack([temp, prec]).transpose(0,1)
        v = self.conv1(tp)
        v = self.batchnorm1(v)
        v = self.activation_fun(v)

        v = self.conv2(v)
        # v = self.activation_fun(v)

        return v.squeeze()


class WideConvRegressiveModel(BaseRegressiveModel):

    def _create_layers(self):
        conv1_nfeatures = 4
        self.conv1 = nn.Conv1d(2, conv1_nfeatures, 7, stride=1, padding_mode='circular', padding=math.floor(7/2))
        self.batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        conv2_nfeatures = 8
        self.conv2 = nn.Conv1d(conv1_nfeatures, conv2_nfeatures, 7, stride=1, padding_mode='circular', padding=math.floor(7/2))
        self.batchnorm2 = nn.BatchNorm1d(conv2_nfeatures)
        conv3_nfeatures = 16
        self.conv3 = nn.Conv1d(conv2_nfeatures, conv3_nfeatures, 7, stride=1, padding_mode='circular', padding=math.floor(7/2))
        self.batchnorm3 = nn.BatchNorm1d(conv3_nfeatures)

        self.conv_end = nn.Conv1d(conv3_nfeatures, 1, 1, stride=1, padding_mode='circular', padding=0)
        self.linear1 = nn.Linear(self.dataset_train.dataset.sample_dim, self.dataset_train.dataset.sample_dim)

    def forward(self, h, temp, prec):
        temp[temp != temp] = 0.
        prec[prec != prec] = 0.
        tp = torch.stack([temp, prec]).transpose(0,1)

        v = self.conv1(tp)
        # v = self.batchnorm1(v)
        v = self.activation_fun(v)

        v = self.conv2(v)
        # v = self.batchnorm2(v)
        v = self.activation_fun(v)

        v = self.conv3(v)
        # v = self.batchnorm3(v)
        v = self.activation_fun(v)
        # v = self.activation_fun(v)
        v = self.conv_end(v)

        return v.squeeze()


class WideStrideConvRegressiveModel(WideConvRegressiveModel):

    def _create_layers(self):
        conv1_nfeatures = 8
        self.conv1 = nn.Conv1d(2, conv1_nfeatures, 7, stride=3, padding_mode='circular', padding=math.floor(7/2))
        self.batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        conv2_nfeatures = 16
        self.conv2 = nn.Conv1d(conv1_nfeatures, conv2_nfeatures, 7, stride=3, padding_mode='circular', padding=math.floor(7/2))
        self.batchnorm2 = nn.BatchNorm1d(conv2_nfeatures)
        conv3_nfeatures = 32
        self.conv3 = nn.Conv1d(conv2_nfeatures, conv3_nfeatures, 7, stride=3, padding_mode='circular', padding=math.floor(7/2))
        self.batchnorm3 = nn.BatchNorm1d(conv3_nfeatures)

        self.conv_end = nn.Conv1d(conv3_nfeatures, 1, 1, stride=1, padding_mode='circular', padding=0)
        self.linear1 = nn.Linear(self.dataset_train.dataset.sample_dim, self.dataset_train.dataset.sample_dim)


class UNet(BaseRegressiveModel):
    def _create_layers(self):

        features = 16
        in_channels = 2
        out_channels = 1

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )


    def forward(self, h, temp, prec):
        temp[temp != temp] = 0.
        prec[prec != prec] = 0.
        tp = torch.stack([temp, prec]).transpose(0,1)

        enc1 = self.encoder1(tp)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return (self.conv(dec1)).squeeze()
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )




class Autoregressive_UNet(BaseRegressiveModel):
    def _create_layers(self):

        features = 16
        in_channels = 2
        out_channels = 1

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        vod_features = 1
        self.conv = nn.Conv1d(
            in_channels=features+vod_features, out_channels=out_channels, kernel_size=1
        )

        features = vod_features
        in_channels = 1
        out_channels = 1

        self.vod_encoder1 = UNet._block(in_channels, features, name="enc1")
        self.vod_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.vod_encoder2 = UNet._block(features, features * 2, name="enc2")
        self.vod_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.vod_encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.vod_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.vod_encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.vod_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.vod_bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.vod_upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.vod_decoder4 = UNet._block((features * 8), features * 8, name="dec4")
        self.vod_upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.vod_decoder3 = UNet._block((features * 4), features * 4, name="dec3")
        self.vod_upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.vod_decoder2 = UNet._block((features * 2), features * 2, name="dec2")
        self.vod_upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.vod_decoder1 = UNet._block(features, features, name="dec1")



    def forward(self, h, temp, prec):
        temp[temp != temp] = 0.
        prec[prec != prec] = 0.
        h[h != h] = 0.
        tp = torch.stack([temp, prec]).transpose(0,1)

        enc1 = self.encoder1(tp)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)



        enc1 = self.vod_encoder1(h[:, None, :])
        enc2 = self.vod_encoder2(self.vod_pool1(enc1))
        enc3 = self.vod_encoder3(self.vod_pool2(enc2))
        enc4 = self.vod_encoder4(self.vod_pool3(enc3))

        bottleneck = self.vod_bottleneck(self.vod_pool4(enc4))

        dec4 = self.vod_upconv4(bottleneck)
        dec4 = self.vod_decoder4(dec4)
        dec3 = self.vod_upconv3(dec4)
        dec3 = self.vod_decoder3(dec3)
        dec2 = self.vod_upconv2(dec3)
        dec2 = self.vod_decoder2(dec2)
        vod_dec1 = self.vod_upconv1(dec2)
        vod_dec1 = self.vod_decoder1(vod_dec1)
        final = torch.cat((dec1, vod_dec1), dim=1)


        return (self.conv(final)).squeeze()
class LSTMRegressiveModel(BaseRegressiveModel):

    def _create_layers(self):
        conv1_nfeatures = 4
        hidden_size=32
        self.num_layers = 3

        self.lstm_1 = nn.LSTM(input_size=2,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.1)

        # self.conv1 = nn.Conv1d(2, conv1_nfeatures, 7, stride=1, padding_mode='circular', padding=math.floor(7/2))
        # self.batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        # conv2_nfeatures = 8
        # self.conv2 = nn.Conv1d(conv1_nfeatures, conv2_nfeatures, 7, stride=1, padding_mode='circular', padding=math.floor(7/2))
        # self.batchnorm2 = nn.BatchNorm1d(conv2_nfeatures)
        # conv3_nfeatures = 16
        # self.conv3 = nn.Conv1d(conv2_nfeatures, conv3_nfeatures, 7, stride=1, padding_mode='circular', padding=math.floor(7/2))
        # self.batchnorm3 = nn.BatchNorm1d(conv3_nfeatures)

        self.conv_end = nn.Conv1d(hidden_size*2, 1, 1, stride=1, padding_mode='circular', padding=0)
        # self.linear1 = nn.Linear(self.dataset_train.dataset.sample_dim, self.dataset_train.dataset.sample_dim)

    def forward(self, h, temp, prec):
        temp[temp != temp] = 0.
        prec[prec != prec] = 0.
        tp = torch.stack([temp, prec]).transpose(0,1)

        v = self.lstm_1(tp.transpose(1, 2))[0]
        # v = self.batchnorm1(v)
        v = self.activation_fun(v)
        v = self.conv_end(v.transpose(1, 2))

        return v.squeeze()
