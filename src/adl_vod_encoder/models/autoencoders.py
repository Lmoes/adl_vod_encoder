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

class BaseModel(pl.LightningModule):
    """
    Minimalistic autoencoder to be used for inheritance for more complex models
    Handles missing values by setting them to 0, and masking them out for loss calculation.
    Therefore, it will still try to predict missing values, but wont be penalized for them.

    Base code structure from these two resources:
    https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0
    https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_autoencoder.py
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None,
                 noise=0., dropout=0.):
        super(BaseModel, self).__init__()
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

        self.e_linear1 = nn.Linear(dataset.sample_dim, encoding_dim)
        self.d_linear1 = nn.Linear(encoding_dim, dataset.sample_dim)

    def encoder(self, h):
        # h = F.dropout(ts, p=0.2)
        h = self.e_linear1(h)
        h = self.activation_fun(h)
        h = h.squeeze()
        return h

    def decoder(self, encoding):
        h = self.d_linear1(encoding)
        return h

    def forward(self, x):
        x[x != x] = 0.
        encoding = self.encoder(x[:, None])
        return encoding

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_nb):
        x = batch[0]
        x = x.to(self.device)
        inputnans = x != x
        h = x + torch.randn_like(x) * self.noise
        h = F.dropout(h, p=self.dropout)
        encoding = self(h)

        x_hat = self.decoder(encoding)
        loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.training_step(batch, batch_nb)
        self.log('val_loss', loss)

        x, temp, _ = batch
        x = x.to(self.device)
        encoding = self(x).cpu()
        # self.log('val_r2', adj_r2(encoding, temp)[0], on_epoch=True)

    # def validation_step_end(self, *args, **kwargs):
    #     encoding = self.encode_ds(self.dataset_val.dataset)
    #     self.log('val_r2', adj_r2(encoding, self.dataset_val.dataset.tempdata)[1], on_epoch=True)


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
            batch_encoding = self(batch[0].to(self.device))
            batch_encoding_list.append(batch_encoding.cpu().detach().numpy())
        encodings = np.concatenate(batch_encoding_list)
        return encodings

    def predict_ds(self, ds):
        """
        Given a dataset, return its predicted VOD
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        batch_ts_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_encoding = self(batch[0].to(self.device))
            batch_ts_hat = self.decoder(batch_encoding)
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
            reconstruction_loss = np.nanmean((ds.data * ds.vod_std - predictions[0] * ds.vod_std)**2,1)
            loss = {'reconstruction_loss_origscale': reconstruction_loss}
        else:

            reconstruction_loss = np.nanmean((ds.data - predictions[0])**2,1)
            loss = {'reconstruction_loss': reconstruction_loss}
        return loss

    @staticmethod
    def cluster_encodings(encodings, nclusters=10, tol=1e-4):
        """
        Cluster the created encoding.
        :param encodings: np.array format [location, latent_variable]
        :param nclusters: int, number of clusters to make
        :param tol: float, tolerance when to stop
        :return:
        """
        cluster_ids_x, _ = kmeans(torch.from_numpy(encodings), nclusters, tol=tol)
        return cluster_ids_x.detach().numpy()

    def predict_td_effect(self, ds, td=1., encoding=None):

        if encoding is None:
            encoding = self.encode_ds(ds)
        temp_adj, encoding_adj = qr_tempadjust(encoding, ds.tempdata, td/ds.temp_std)
        ts_hat = self.decoder(torch.from_numpy(encoding_adj).to(self.device))
        print("Adjusted r2 r2: {}/{}".format(*adj_r2(encoding, ds.tempdata)))
        for i in range(encoding.shape[-1]):
            coef, _ = pearsonr(encoding[:,i], ds.tempdata)
            print("coef encoding {}: {}".format(i, coef**2))
        #
        # batch_ts_hat_list = []
        # for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
        #     batch_encoding = self(batch[0].to(self.device))
        #     adjusted_encoding = self.adjust_encoding(batch_encoding, td)
        #     batch_ts_hat = self.decoder(adjusted_encoding)
        #     batch_ts_hat_list.append(batch_ts_hat.cpu().detach().numpy())
        # ts_hats = np.concatenate(batch_ts_hat_list)
        return ts_hat.detach().numpy()

# class BaseTempModeler(object):
#     def predict_td_effect(self, ds, td=1., encoding=None):
#
#         if encoding is None:
#             encoding = self.encode_ds(ds)
#         temp = ds.tempdata * ds.temp_std + ds.temp_mean
#         temp_adj, encoding_adj = qr_tempadjust(encoding, temp, td)
#         ts_hat = self.decoder(torch.from_numpy(encoding_adj).to(self.device))
#         #
#         # batch_ts_hat_list = []
#         # for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
#         #     batch_encoding = self(batch[0].to(self.device))
#         #     adjusted_encoding = self.adjust_encoding(batch_encoding, td)
#         #     batch_ts_hat = self.decoder(adjusted_encoding)
#         #     batch_ts_hat_list.append(batch_ts_hat.cpu().detach().numpy())
#         # ts_hats = np.concatenate(batch_ts_hat_list)
#         return ts_hat.detach().numpy()
#
#     def adjust_encoding(self, encoding, td=0.):
#         raise NotImplementedError
#
#     def
#
class ShallowConvAutoencoder(BaseModel):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(ShallowConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
        conv1_width = 7
        conv1_nfeatures = 32
        self.encoder = nn.Sequential(
            nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=1, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.Flatten(),
            nn.Linear((dataset.sample_dim - conv1_width + 1) * conv1_nfeatures, encoding_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, (dataset.sample_dim - conv1_width + 1) * conv1_nfeatures),
            nn.ReLU(),
            Reshape((-1, conv1_nfeatures, (dataset.sample_dim - conv1_width + 1))),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width),
            Squeeze()
        )

class MonthlyShallowConvAutoencoder(BaseModel):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None, noise=0., dropout=0.):
        super(MonthlyShallowConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr, activation_fun, noise, dropout)
        conv1_width = 3
        conv1_nfeatures = 16
        self.encoder = nn.Sequential(
            nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=1, padding_mode='circular'),
            self.activation_fun,
            nn.BatchNorm1d(conv1_nfeatures),
            self.activation_fun,
            nn.Linear((dataset.sample_dim - conv1_width + 1) * conv1_nfeatures, encoding_dim),
            self.activation_fun
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, (dataset.sample_dim - conv1_width + 1) * conv1_nfeatures),
            self.activation_fun,
            Reshape((-1, conv1_nfeatures, (dataset.sample_dim - conv1_width + 1))),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width),
            Squeeze()
        )


class DeepConvAutoencoder(BaseModel):
    """
    Convolutional autoencoder. Similar to https://arxiv.org/abs/2002.03624v1
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None):
        super(DeepConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr, activation_fun)
        conv1_width = 3
        conv2_width = 4
        conv1_nfeatures = 16

        # encoder layers
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=2, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv1_nfeatures * 2, kernel_size=conv2_width, stride=3)
        self.e_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures * 2)
        self.e_linear1 = nn.Linear(conv1_nfeatures * 2 * 243, 64)
        self.e_linear2 = nn.Linear(64, encoding_dim)
        # decoder layers
        self.d_linear2 = nn.Linear(encoding_dim, 64)
        self.d_linear1 = nn.Linear(64, conv1_nfeatures * 2 * 243)
        self.d_deconv2 = nn.ConvTranspose1d(conv1_nfeatures * 2, conv1_nfeatures, kernel_size=conv2_width, stride=3)
        # self.d_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures)
        self.d_deconv1 = nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width, stride=2, padding_mode='zeros')

    def encoder(self, h):
        # h = h + torch.randn_like(h) * 0.1
        # h = F.dropout(h, p=0.2)
        h = self.e_conv1(h)
        h = self.activation_fun(h)
        h = self.e_batchnorm1(h)
        h = self.e_conv2(h)
        h = self.activation_fun(h)
        h = self.e_batchnorm2(h)
        h = h.view(-1, h.shape[1] * 243)
        h = self.e_linear1(h)
        h = self.activation_fun(h)
        h = self.e_linear2(h)
        # h = torch.sigmoid(h)
        h = self.activation_fun(h)
        h = h.squeeze()
        return h

    def decoder(self, encoding):
        h = self.d_linear2(encoding)
        h = self.activation_fun(h)
        h = self.d_linear1(h)
        h = self.activation_fun(h)
        h = h.view(-1, self.d_deconv2.in_channels, 243)
        h = self.d_deconv2(h)
        # h = self.d_batchnorm2(h)
        h = self.activation_fun(h)
        h = self.d_deconv1(h)
        h = h.squeeze()
        return h


class VeryDeepConvAutoencoder(BaseModel):
    """
    Convolutional autoencoder. Similar to https://arxiv.org/abs/2002.03624v1
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None):
        super(VeryDeepConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr, activation_fun)
        conv1_width = 3
        conv2_width = 4
        conv3_width = 3
        conv1_nfeatures = 16
        conv2_nfeatures = 32
        conv3_nfeatures = 64
        linear_features = 128

        # encoder layers
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=2, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv2_nfeatures, kernel_size=conv2_width, stride=3)
        self.e_batchnorm2 = nn.BatchNorm1d(conv2_nfeatures)

        self.e_conv3 = nn.Conv1d(conv2_nfeatures, conv3_nfeatures, kernel_size=conv3_width, stride=2)
        self.e_batchnorm3 = nn.BatchNorm1d(conv3_nfeatures)

        self.e_linear1 = nn.Linear(conv3_nfeatures * 121, linear_features)
        self.e_linear2 = nn.Linear(linear_features, encoding_dim)
        # decoder layers
        self.d_linear2 = nn.Linear(encoding_dim, linear_features)
        self.d_linear1 = nn.Linear(linear_features, conv3_nfeatures * 121)

        self.d_deconv3 = nn.ConvTranspose1d(conv3_nfeatures, conv2_nfeatures, kernel_size=conv3_width, stride=2)
        self.d_batchnorm3 = nn.BatchNorm1d(conv2_nfeatures)

        self.d_deconv2 = nn.ConvTranspose1d(conv2_nfeatures, conv1_nfeatures, kernel_size=conv2_width, stride=3)
        self.d_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures)
        self.d_deconv1 = nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width, stride=2, padding_mode='zeros')

    def encoder(self, ts):
        # h = h + torch.randn_like(h) * 0.1
        # h = F.dropout(h, p=0.2)
        h = self.e_conv1(ts)
        h = self.activation_fun(h)
        h = self.e_batchnorm1(h)
        h = self.e_conv2(h)
        h = self.activation_fun(h)
        h = self.e_batchnorm2(h)

        h = self.e_conv3(h)
        h = self.activation_fun(h)
        h = self.e_batchnorm3(h)

        h = h.view(-1, h.shape[1] * 121)
        h = self.e_linear1(h)
        h = self.activation_fun(h)
        h = self.e_linear2(h)
        # h = torch.sigmoid(h)
        h = self.activation_fun(h)
        h = h.squeeze()
        return h

    def decoder(self, encoding):
        h = self.d_linear2(encoding)
        h = self.activation_fun(h)
        h = self.d_linear1(h)
        h = self.activation_fun(h)
        h = h.view(-1, self.d_deconv3.in_channels, 121)

        h = self.d_deconv3(h)
        h = self.d_batchnorm3(h)
        h = self.activation_fun(h)

        h = self.d_deconv2(h)
        h = self.d_batchnorm2(h)
        h = self.activation_fun(h)

        h = self.d_deconv1(h)
        h = h.squeeze()
        return h



class MonthlyVeryDeepConvAutoencoder(BaseModel):
    """
    Convolutional autoencoder. Similar to https://arxiv.org/abs/2002.03624v1
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None):
        super(MonthlyVeryDeepConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr, activation_fun)
        conv1_width = 3
        conv2_width = 12
        conv3_width = 10
        conv1_nfeatures = 16
        conv2_nfeatures = 32
        conv3_nfeatures = 64
        linear_features = 64
        self.magic_num = 314
        # encoder layers
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=1, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv2_nfeatures, kernel_size=conv2_width, stride=1)
        self.e_batchnorm2 = nn.BatchNorm1d(conv2_nfeatures)

        self.e_conv3 = nn.Conv1d(conv2_nfeatures, conv3_nfeatures, kernel_size=conv3_width, stride=1)
        self.e_batchnorm3 = nn.BatchNorm1d(conv3_nfeatures)

        self.e_linear1 = nn.Linear(conv3_nfeatures * self.magic_num, linear_features)
        self.e_linear2 = nn.Linear(linear_features, encoding_dim)
        # decoder layers
        self.d_linear2 = nn.Linear(encoding_dim, linear_features)
        self.d_linear1 = nn.Linear(linear_features, conv3_nfeatures * self.magic_num)

        self.d_deconv3 = nn.ConvTranspose1d(conv3_nfeatures, conv2_nfeatures, kernel_size=conv3_width, stride=1)
        self.d_batchnorm3 = nn.BatchNorm1d(conv2_nfeatures)

        self.d_deconv2 = nn.ConvTranspose1d(conv2_nfeatures, conv1_nfeatures, kernel_size=conv2_width, stride=1)
        self.d_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures)
        self.d_deconv1 = nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width, stride=1, padding_mode='zeros')

    def encoder(self, h):
        h = h + torch.randn_like(h) * 0.1
        h = F.dropout(h, p=0.5)
        h = self.e_conv1(h)
        h = self.activation_fun(h)
        h = self.e_batchnorm1(h)
        h = self.e_conv2(h)
        h = self.activation_fun(h)
        h = self.e_batchnorm2(h)

        h = self.e_conv3(h)
        h = self.activation_fun(h)
        h = self.e_batchnorm3(h)

        h = h.view(-1, h.shape[1] * self.magic_num)
        h = self.e_linear1(h)
        h = self.activation_fun(h)
        h = self.e_linear2(h)
        # h = torch.sigmoid(h)
        h = self.activation_fun(h)
        h = h.squeeze()
        return h

    def decoder(self, encoding):
        h = self.d_linear2(encoding)
        h = self.activation_fun(h)
        h = self.d_linear1(h)
        h = self.activation_fun(h)
        h = h.view(-1, self.d_deconv3.in_channels, self.magic_num)

        h = self.d_deconv3(h)
        h = self.d_batchnorm3(h)
        h = self.activation_fun(h)

        h = self.d_deconv2(h)
        h = self.d_batchnorm2(h)
        h = self.activation_fun(h)

        h = self.d_deconv1(h)
        h = h.squeeze()
        return h


class Monthly4DeepConvAutoencoder(BaseModel):
    """
    Convolutional autoencoder. Similar to https://arxiv.org/abs/2002.03624v1
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None, noise=0,
                 dropout=0.):
        super(Monthly4DeepConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr, activation_fun,
                                                          noise, dropout)
        conv1_width = 3
        conv2_width = 3
        conv3_width = 6
        conv4_width = 12
        conv1_nfeatures = 8
        conv2_nfeatures = 16
        conv3_nfeatures = 32
        conv4_nfeatures = 64
        linear_features = 64
        self.magic_num = 316
        # encoder layers
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=1, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)

        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv2_nfeatures, kernel_size=conv2_width, stride=1)
        self.e_batchnorm2 = nn.BatchNorm1d(conv2_nfeatures)

        self.e_conv3 = nn.Conv1d(conv2_nfeatures, conv3_nfeatures, kernel_size=conv3_width, stride=1)
        self.e_batchnorm3 = nn.BatchNorm1d(conv3_nfeatures)

        self.e_conv4 = nn.Conv1d(conv3_nfeatures, conv4_nfeatures, kernel_size=conv4_width, stride=1)
        self.e_batchnorm4 = nn.BatchNorm1d(conv4_nfeatures)

        self.e_linear1 = nn.Linear(conv4_nfeatures * self.magic_num, encoding_dim)
        # self.e_linear2 = nn.Linear(linear_features, encoding_dim)


        # decoder layers
        # self.d_linear2 = nn.Linear(encoding_dim, linear_features)
        self.d_linear1 = nn.Linear(encoding_dim, conv4_nfeatures * self.magic_num)

        self.d_deconv4 = nn.ConvTranspose1d(conv4_nfeatures, conv3_nfeatures, kernel_size=conv4_width, stride=1,  padding_mode='zeros')
        self.d_batchnorm4 = nn.BatchNorm1d(conv3_nfeatures)

        self.d_deconv3 = nn.ConvTranspose1d(conv3_nfeatures, conv2_nfeatures, kernel_size=conv3_width, stride=1,  padding_mode='zeros')
        self.d_batchnorm3 = nn.BatchNorm1d(conv2_nfeatures)

        self.d_deconv2 = nn.ConvTranspose1d(conv2_nfeatures, conv1_nfeatures, kernel_size=conv2_width, stride=1)
        self.d_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures)
        self.d_deconv1 = nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width, stride=1, padding_mode="zeros")

    def encoder(self, h):
        # print(h.shape)
        # # h = h + torch.randn_like(h) * 0.3
        # h = F.dropout(h, p=0.1)
        h = self.e_conv1(h)
        # print(h.shape)
        h = self.activation_fun(h)
        h = self.e_batchnorm1(h)
        h = self.e_conv2(h)
        # print(h.shape)
        h = self.activation_fun(h)
        h = self.e_batchnorm2(h)

        h = self.e_conv3(h)
        # print(h.shape)
        h = self.activation_fun(h)
        h = self.e_batchnorm3(h)

        h = self.e_conv4(h)
        # print(h.shape)
        h = self.activation_fun(h)
        h = self.e_batchnorm4(h)

        h = h.view(-1, h.shape[1] * self.magic_num)
        h = self.e_linear1(h)
        # h = self.activation_fun(h)
        # h = self.e_linear2(h)
        # h = torch.sigmoid(h)
        h = self.activation_fun(h)
        h = h.squeeze()
        return h

    def decoder(self, encoding):
        # h = self.d_linear2(encoding)
        # h = self.activation_fun(h)
        h = self.d_linear1(encoding)
        h = self.activation_fun(h)
        h = h.view(-1, self.d_deconv4.in_channels, self.magic_num)

        h = self.d_deconv4(h)
        h = self.d_batchnorm4(h)
        h = self.activation_fun(h)

        h = self.d_deconv3(h)
        h = self.d_batchnorm3(h)
        h = self.activation_fun(h)

        h = self.d_deconv2(h)
        h = self.d_batchnorm2(h)
        h = self.activation_fun(h)

        h = self.d_deconv1(h)
        h = h.squeeze()
        return h
class MontlyLstmAutencoder(BaseModel):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001, activation_fun=None, noise=0,
                 dropout=0.):
        super(MontlyLstmAutencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr, activation_fun,
                                                          noise, dropout)
        # self.magic_num = 314
        self.num_layers = 3
        hidden_size=4
        self.e_lstm_1 = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True, bidirectional=True, num_layers=self.num_layers, dropout=0.1)
        # self.e_linear2 = nn.Linear(self.num_layers*2*hidden_size, encoding_dim)
        # self.d_linear2 = nn.Linear(encoding_dim, self.num_layers*2*hidden_size)
        self.d_lstm_1 = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True, bidirectional=True, num_layers=self.num_layers, dropout=0.1)

        self.out = nn.Conv1d(8, 1, 1)

        conv1_width = 3
        conv2_width = 3
        conv3_width = 6
        conv4_width = 12
        conv1_nfeatures = 8
        conv2_nfeatures = 16
        conv3_nfeatures = 32
        conv4_nfeatures = 64
        linear_features = 64
        self.magic_num = 316
        # encoder layers
        self.e_conv1 = nn.Conv1d(8, conv1_nfeatures, conv1_width, stride=1)
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)

        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv2_nfeatures, kernel_size=conv2_width, stride=1)
        self.e_batchnorm2 = nn.BatchNorm1d(conv2_nfeatures)

        self.e_conv3 = nn.Conv1d(conv2_nfeatures, conv3_nfeatures, kernel_size=conv3_width, stride=1)
        self.e_batchnorm3 = nn.BatchNorm1d(conv3_nfeatures)

        self.e_conv4 = nn.Conv1d(conv3_nfeatures, conv4_nfeatures, kernel_size=conv4_width, stride=1)
        self.e_batchnorm4 = nn.BatchNorm1d(conv4_nfeatures)

        self.e_linear1 = nn.Linear(conv4_nfeatures * self.magic_num, encoding_dim)
        # self.e_linear2 = nn.Linear(linear_features, encoding_dim)


        # decoder layers
        # self.d_linear2 = nn.Linear(encoding_dim, linear_features)
        self.d_linear1 = nn.Linear(encoding_dim, conv4_nfeatures * self.magic_num)

        self.d_deconv4 = nn.ConvTranspose1d(conv4_nfeatures, conv3_nfeatures, kernel_size=conv4_width, stride=1,  padding_mode='zeros')
        self.d_batchnorm4 = nn.BatchNorm1d(conv3_nfeatures)

        self.d_deconv3 = nn.ConvTranspose1d(conv3_nfeatures, conv2_nfeatures, kernel_size=conv3_width, stride=1,  padding_mode='zeros')
        self.d_batchnorm3 = nn.BatchNorm1d(conv2_nfeatures)

        self.d_deconv2 = nn.ConvTranspose1d(conv2_nfeatures, conv1_nfeatures, kernel_size=conv2_width, stride=1)
        self.d_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures)
        self.d_deconv1 = nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width, stride=1, padding_mode="zeros")

    def encoder(self, ts):
        h, (_, _) = self.e_lstm_1(ts.transpose(1,2))
        h = self.activation_fun(h)
        h = self.e_conv1(h.transpose(1,2))

        h = self.activation_fun(h)
        h = self.e_batchnorm1(h)
        h = self.e_conv2(h)
        # print(h.shape)
        h = self.activation_fun(h)
        h = self.e_batchnorm2(h)

        h = self.e_conv3(h)
        # print(h.shape)
        h = self.activation_fun(h)
        h = self.e_batchnorm3(h)

        h = self.e_conv4(h)
        # print(h.shape)
        h = self.activation_fun(h)
        h = self.e_batchnorm4(h)

        h = h.view(-1, h.shape[1] * self.magic_num)
        h = self.e_linear1(h)
        h = self.activation_fun(h)
        h = h.squeeze()
        return h

    def decoder(self, encoding):
        # h = self.d_linear2(encoding)
        # h = self.activation_fun(h)
        h = self.d_linear1(encoding)
        h = self.activation_fun(h)
        h = h.view(-1, self.d_deconv4.in_channels, self.magic_num)

        h = self.d_deconv4(h)
        h = self.d_batchnorm4(h)
        h = self.activation_fun(h)

        h = self.d_deconv3(h)
        h = self.d_batchnorm3(h)
        h = self.activation_fun(h)

        h = self.d_deconv2(h)
        h = self.d_batchnorm2(h)
        h = self.activation_fun(h)

        h = self.d_deconv1(h)

        h, (_, _) = self.d_lstm_1(h.transpose(1, 2))
        # h = h.squeeze()
        #
        # h = self.d_linear2(encoding)
        # h = self.activation_fun(h)
        # n_samples = self.dataset_train.dataset.sample_dim
        # h = h.view(-1, 1, self.num_layers*2*4).repeat(1,n_samples,1)
        # h, (_, _) = self.d_lstm_1(h)
        h = self.out(h.transpose(1,2)).squeeze()
        return h

    # def forward(self, x):
    #     x[x != x] = 0.
    #     encoding = self.encoder(x[:, None])
    #     return encoding
    #
    # def training_step(self, batch, batch_nb):
    #     x = batch[0]
    #     x = x.to(self.device)
    #     inputnans = x != x
    #     h = x + torch.randn_like(x) * self.noise
    #     h = F.dropout(h, p=self.dropout)
    #     encoding = self(h)
    #
    #     x_hat = self.decoder(encoding)
    #     loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
    #     return loss


class BaseTempPrecAutoencoder(BaseModel):
    """
    Minimalistic autoencoder that also predicts the temperature and precipitation as side tasks
    """
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
        loss = reconstruction_loss*0.8 + p_loss*0.2 # + t_loss*0.1
        return loss

    # def validation_step(self, batch, batch_nb):
    #     loss = self.training_step(batch, batch_nb)
    #     self.log('val_loss', loss, on_epoch=True)
    #     x, temp, _ = batch
    #     x = x.to(self.device)
    #     encoding = self(x).cpu()
    #     self.log('val_r2', adj_r2(encoding, temp)[1], on_epoch=True)

        # print("r2/Adjusted r2: {}/{}".format(*adj_r2(encoding, temp)))


    def predict_ds(self, ds):
        batch_ts_hat_list = []
        batch_t_hat_list = []
        batch_p_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):

            batch_encoding = self(batch[0].to(self.device))
            batch_x_hat = self.decoder(batch_encoding)
            batch_t_hat = self.temppredictor(batch_encoding)
            batch_p_hat = self.precpredictor(batch_encoding)

            batch_ts_hat_list.append(batch_x_hat.cpu().detach().numpy())
            batch_t_hat_list.append(batch_t_hat.cpu().detach().numpy())
            batch_p_hat_list.append(batch_p_hat.cpu().detach().numpy())
        x_hats = np.concatenate(batch_ts_hat_list)
        t_hats = np.concatenate(batch_t_hat_list)
        p_hats = np.concatenate(batch_p_hat_list)
        return x_hats, t_hats, p_hats

    def predict_td_effect(self, ds, td=1., encoding=None):

        if encoding is None:
            encoding = self.encode_ds(ds)
        temp = ds.tempdata# * ds.temp_std + ds.temp_mean
        temp_adj, encoding_adj = qr_tempadjust(encoding, temp, td/ds.temp_std)
        encoding_adj = torch.from_numpy(encoding_adj).to(self.device)
        ts_hat = self.decoder(encoding_adj)
        t_hat = self.temppredictor(encoding_adj)
        p_hat = self.precpredictor(encoding_adj)

        print("r2/Adjusted r2: {}/{}".format(*adj_r2(encoding, temp)))
        for i in range(encoding.shape[-1]):
            coef, _ = pearsonr(encoding[:,i], temp)
            print("coef encoding {}: {}".format(i, coef**2))

        return ts_hat.detach().numpy(), t_hat.detach().numpy(), p_hat.detach().numpy()



    def loss_all(self, predictions, ds, origscale=False):
        """
        Calculate the loss of a set of predictions
        :param predictions: list of np arrays [vod_predicted, temp_predicted, prec_predicted]
            predicted vod, temperature and precipitation values
        :param ds:
        :param origscale:
        :return:
        """
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
                                           nn.ReLU(),
                                           nn.Linear(auxpreddim, 1))
        self.precpredictor = nn.Sequential(nn.Linear(encoding_dim, auxpreddim),
                                           nn.ReLU(),
                                           nn.Linear(auxpreddim, 1))

        conv1_width = 7
        conv1_nfeatures = 32
        self.encoder = nn.Sequential(
            nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=1, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.Flatten(),
            nn.Linear((dataset.sample_dim - conv1_width + 1) * conv1_nfeatures, encoding_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, (dataset.sample_dim - conv1_width + 1) * conv1_nfeatures),
            nn.ReLU(),
            Reshape((-1, conv1_nfeatures, (dataset.sample_dim - conv1_width + 1))),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width),
            Squeeze()
        )


class DeepConvTempPrecAutoencoder(BaseTempPrecAutoencoder):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(DeepConvTempPrecAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
        auxpreddim = 8

        self.temppredictor = nn.Sequential(nn.Linear(encoding_dim, auxpreddim),
                                           nn.ReLU(),
                                           nn.Linear(auxpreddim, 1))
        self.precpredictor = nn.Sequential(nn.Linear(encoding_dim, auxpreddim),
                                           nn.ReLU(),
                                           nn.Linear(auxpreddim, 1))
        conv1_width = 3
        conv2_width = 4
        conv1_nfeatures = 16

        # encoder layers
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=2, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv1_nfeatures * 2, kernel_size=conv2_width, stride=3)
        self.e_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures * 2)
        self.e_linear1 = nn.Linear(conv1_nfeatures * 2 * 243, 64)
        self.e_linear2 = nn.Linear(64, encoding_dim)
        # decoder layers
        self.d_linear2 = nn.Linear(encoding_dim, 64)
        self.d_linear1 = nn.Linear(64, conv1_nfeatures * 2 * 243)
        self.d_deconv2 = nn.ConvTranspose1d(conv1_nfeatures * 2, conv1_nfeatures, kernel_size=conv2_width, stride=3)
        self.d_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures)
        self.d_deconv1 = nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width, stride=2, padding_mode='zeros')

    def encoder(self, h):
        # h = F.dropout(ts, p=0.2)
        h = self.e_conv1(h)
        h = F.elu(h)
        h = self.e_batchnorm1(h)

        h = self.e_conv2(h)
        h = F.elu(h)
        h = self.e_batchnorm2(h)
        h = h.view(-1, h.shape[1] * 243)
        h = self.e_linear1(h)
        h = F.elu(h)
        h = self.e_linear2(h)
        h = torch.sigmoid(h)

        h = h.squeeze()
        return h

    def decoder(self, encoding):
        h = self.d_linear2(encoding)
        h = F.elu(h)
        h = self.d_linear1(h)
        h = F.elu(h)
        h = h.view(-1, self.d_deconv2.in_channels, 243)
        h = self.d_deconv2(h)
        h = self.d_batchnorm2(h)
        h = F.elu(h)
        h = self.d_deconv1(h)
        h = h.squeeze()
        return h


class SplitYearAutoencoder(BaseModel):
    """
    Autoencoder where each year is encoded/decoded independently.

    The basic idea behind this encoder is that the climate is static over time (which it mostly is for 30 years).
     We now split the time series of each location by year, and encode each year independently. If all encodings of a
     time series are more similar to each other than to the encodings of the other time series within the batch, the
     model is rewarded, and penalized other wise.

     Produces one encoding per year. We can cluster all years independently and then take a majority vote of all
      years. The agreement between years is an indicator for how sure the model is with its classification.

    The model is very minimalistic as this class is mostly there to be inherited from for more complex models.

    """
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(SplitYearAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)

        self.encoder = nn.Sequential(
            View((-1, 52)),
            nn.Linear(52, encoding_dim),
            View((-1, 28, encoding_dim)),

        )
        self.decoder = nn.Sequential(
            View((-1, encoding_dim)),
            nn.Linear(encoding_dim, 52),
            View((-1, 52*28)),
        )

    def forward(self, x):
        x[x != x] = 0.
        encoding = self.encoder(x[:, None])
        return encoding

    def training_step(self, batch, batch_nb):
        x = batch.to(self.device)
        inputnans = x != x
        encoding = self(x)
        disp_loss = -normalized_scatter_ratio(encoding)

        x_hat = self.decoder(encoding)
        loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        loss = loss * (2 + disp_loss)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.training_step(batch, batch_nb)
        self.log('val_loss', loss)

    def encode_ds(self, ds):
        """
        Given a dataset, encode it
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        batch_encoding_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_encoding = self(batch.to(self.device))
            batch_encoding_list.append(batch_encoding.cpu().detach().numpy())
        encodings = np.concatenate(batch_encoding_list)
        return encodings

    @staticmethod
    def cluster_encodings(encodings, nclusters=10, tol=1e-4):
        cluster_ids_x, _ = kmeans(torch.from_numpy(encodings).view(-1, encodings.shape[-1]), nclusters, tol=tol)
        cluster_ids_x = cluster_ids_x.view(-1, 28)
        return cluster_ids_x.detach().numpy()


class SplitYearConvAutoencoder(SplitYearAutoencoder):
    """
    Same idea as class it inherits from, but with a deep convolutional setup
    """
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(SplitYearConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
        conv1_nfeatures = 32
        conv2_nfeatures = 64
        self.encoder = nn.Sequential(
            View((-1, 1, 52)),
            nn.Conv1d(1, conv1_nfeatures, 3),
            nn.BatchNorm1d(conv1_nfeatures),
            nn.ReLU(),
            nn.Conv1d(conv1_nfeatures, conv2_nfeatures, 3),
            nn.BatchNorm1d(conv2_nfeatures),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.ReLU(),
            nn.Conv1d(conv2_nfeatures, conv2_nfeatures, 3),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.ReLU(),
            View((-1, 1, conv2_nfeatures * 4)),
            nn.Linear(conv2_nfeatures * 4, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.Sigmoid(),
            View((-1, 28, encoding_dim)),
        )

        self.decoder = nn.Sequential(
            View((-1, encoding_dim)),
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, conv2_nfeatures * 46),
            nn.ReLU(),
            View((-1, conv2_nfeatures, 46)),
            nn.ConvTranspose1d(conv2_nfeatures, conv2_nfeatures, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(conv2_nfeatures, conv1_nfeatures, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(conv1_nfeatures, 1, 3),
            View((-1, 52 * 28)),
        )