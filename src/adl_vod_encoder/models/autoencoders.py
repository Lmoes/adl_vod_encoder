import torch
from torch import nn, optim, rand, reshape, from_numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
import numpy as np
from src.adl_vod_encoder.models.layers import Squeeze, Reshape, View
from src.adl_vod_encoder.models.validation_metrics import normalized_scatter_ratio
import pandas as pd
# from kmeans_pytorch import kmeans
from scipy.interpolate import interp1d
import xarray as xr
# import pyro
# import pyro.contrib.gp as gp
# import pyro.distributions as dist
# import gpytorch
import matplotlib.pyplot as plt


class BaseModel(pl.LightningModule):
    """
    Minimalistic autoencoder to be used for inheritance by more complex models
    Handles missing values by setting them to 0, and masking them out for loss calculation.
    Therefore it will still try to predict missing values, but wont be penalized for them.

    Base code structure from these two resources:
    https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0
    https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_autoencoder.py
    """

    def __init__(self, dataset, train_size_frac=0.7, batch_size=512, lr=0.001, num_workers=0, nan_fillvalue=0.):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        trainsize = int(len(dataset)*train_size_frac)
        valsize = len(dataset) - trainsize
        self.num_workers = num_workers
        self.nan_fillvalue = nan_fillvalue
        self.dataset_train, self.dataset_val = random_split(dataset, [trainsize, valsize])

        # self.e_linear1 = nn.Linear(dataset.sample_dim, encoding_dim)
        # self.d_linear1 = nn.Linear(encoding_dim, dataset.sample_dim)

    def encoder(self, ts):
        h = self.e_linear1(ts)
        h = F.sigmoid(h)
        h = h.squeeze()
        return h

    def decoder(self, encoding):
        h = self.d_linear1(encoding)
        return h

    def forward(self, x):
        x[x != x] = self.nan_fillvalue
        encoding = self.encoder(x[:, None])
        return encoding

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_nb):
        x = batch[0]
        x = x.to(self.device)
        inputnans = x != x
        encoding = self(x)

        x_hat = self.decoder(encoding)
        loss = F.mse_loss(x_hat[~inputnans], x[~inputnans])
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.training_step(batch, batch_nb)
        self.log('val_loss', loss)

    def train_dataloader(self):
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
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


    def loss_per_gaplength(self, predictions, ds, appendname=""):
        def get_gapdist(x, maxgap=10):
            isnan = np.isnan(x)
            temp = (~isnan).astype(int)

            conv_list = []
            for i in range(maxgap):
                conv_list.append(np.convolve(temp, np.ones(i * 2 + 1), "same"))

            gapdist = ((np.stack(conv_list) == 0).astype(int) * np.arange(1, maxgap+1)[:, None]).max(0)
            return gapdist

        rmsd = (ds.data - predictions) ** 2
        temp = ds.da.values[:, ds.tslocs].T.astype(np.float32)
        temp[~ds.trainidx[:, ds.tslocs].T] = np.nan
        k = np.apply_along_axis(get_gapdist, 1, temp)
        groups = xr.DataArray(rmsd).groupby(xr.DataArray(k))
        quants = groups.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

        quants.name = "loss_gaplength_linear_{}".format(appendname)
        quants = quants.rename({"group": "gaplength"})
        return quants

    def loss_all(self, predictions, ds, origscale=False):
        """
        Given a dataset, return the loss of it
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        if origscale:
            reconstruction_loss = np.nanmean((ds.data * ds.vod_std - predictions * ds.vod_std)**2,1)
            loss = {'reconstruction_loss_origscale': reconstruction_loss}

        else:
            rmsd = (ds.data - predictions)**2
            reconstruction_loss = np.nanmean(rmsd,1)
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

def ceildiv(a, b):
    return -(-a // b)

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv1d(in_c, out_c, 3, stride=1, padding=1),
        nn.ELU(inplace=True),
        nn.Conv1d(out_c, out_c, 3, stride=1, padding=1),
        nn.ELU(inplace=True)
    )
    return conv


def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    deltaleft = delta // 2
    deltaright = ceildiv(delta, 2)

    return tensor[:, :, deltaleft:tensor_size - deltaright]



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        conv1_width = 3
        conv2_width = 4
        conv1_nfeatures = 16
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=2, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv1_nfeatures * 2, kernel_size=conv2_width, stride=3)
        self.e_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures * 2)
        self.e_linear1 = nn.Linear(conv1_nfeatures * 2 * 242, 64)
        self.e_linear2 = nn.Linear(64, 3)

    def forward(self, x):
        e2 = self.e_conv1(x)
        e3 = F.elu(e2)
        e4 = self.e_batchnorm1(e3)
        e5 = self.e_conv2(e4)
        e6 = F.elu(e5)
        e7 = self.e_batchnorm2(e6)
        e8 = e7.view(e7.shape[0], -1)
        e9 = self.e_linear1(e8)
        e10 = F.elu(e9)
        e11 = self.e_linear2(e10)
        # encoding = torch.sigmoid(e11)
        return e11



class GPgapfiller(BaseModel):
    def __init__(self, dataset, train_size_frac=0.7, batch_size=256, lr=0.001):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        trainsize = int(len(dataset)*train_size_frac)
        valsize = len(dataset) - trainsize
        self.dataset_train, self.dataset_val = random_split(dataset, [trainsize, valsize])
        conv1_width = 3
        conv2_width = 4
        conv1_nfeatures = 16
        # encoder layers
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=2, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv1_nfeatures * 2, kernel_size=conv2_width, stride=3)
        self.e_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures * 2)
        self.e_linear1 = nn.Linear(conv1_nfeatures * 2 * 242, 64)
        self.e_linear2 = nn.Linear(64, 3)
        # decoder layers

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2048]))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3, batch_shape=torch.Size([2048])), batch_shape=torch.Size([2048]))

    def forward(self, x):
        x[x != x] = 0.
        e2 = self.e_conv1(x[:, None])
        e3 = F.elu(e2)
        e4 = self.e_batchnorm1(e3)
        e5 = self.e_conv2(e4)
        e6 = F.elu(e5)
        e7 = self.e_batchnorm2(e6)
        e8 = e7.view(e7.shape[0], -1)
        e9 = self.e_linear1(e8)
        e10 = F.elu(e9)
        e11 = self.e_linear2(e10)
        encoding = torch.sigmoid(e11)

        mean_x = self.mean_module(encoding[:, 0, None, None])
        covar_x = self.covar_module(encoding[:, 1, None, None])
        out = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        kernel = gp.kernels.RBF(input_dim=1, variance=encoding[:, 0],
                                lengthscale=encoding[:, 1])
        kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.),
                                lengthscale=torch.tensor(10.))
        gpr = gp.models.GPRegression(torch.arange(0., x.shape[1]), x, kernel, noise=encoding[:, 2])


        d1 = self.d_linear2(encoding)
        d2 = F.elu(d1)
        d3 = self.d_linear1(d2)
        d4 = F.elu(d3)
        d5 = d4.view(-1, self.d_deconv2.in_channels, 243)
        d6 = self.d_deconv2(d5)
        d7 = self.d_batchnorm2(d6)
        d8 = F.elu(d7)
        d9 = self.d_deconv1(d8)
        d10 = d9.squeeze()
        return d10

class BaseConvGapFiller2(BaseModel):
    def __init__(self, dataset, train_size_frac=0.7, batch_size=256, lr=0.001):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        trainsize = int(len(dataset)*train_size_frac)
        valsize = len(dataset) - trainsize
        self.dataset_train, self.dataset_val = random_split(dataset, [trainsize, valsize])
        self.max_pool_2 = nn.MaxPool1d(2, 2)
        self.e_conv_1 = double_conv(1, 8)
        self.e_conv_2 = double_conv(16, 32)
        self.e_conv_3 = double_conv(32, 64)

        self.d_trans_1 = nn.ConvTranspose1d(64, 32, 2, 2)
        self.d_conv_1 = double_conv(64, 32)

        self.d_trans_2 = nn.ConvTranspose1d(32, 16, 2, 2)
        self.d_conv_2 = double_conv(32, 16)

        self.out = nn.Conv1d(16, 1, 1, padding=1)

    def forward(self, x):
        x[x != x] = self.nan_fillvalue
        x1 = self.e_conv_1(x[:, None])
        x2 = self.max_pool_2(x1)

        x3 = self.e_conv_2(x2)
        x4 = self.max_pool_2(x3)

        x5 = self.e_conv_3(x4)

        x6 = self.d_trans_1(x5)
        x7 = crop_tensor(x3, x6)
        x8 = self.d_conv_1(torch.cat([x6, x7], 1))

        x9 = self.d_trans_2(x8)
        x10 = crop_tensor(x1, x9)
        x11 = self.d_conv_2(torch.cat([x9, x10], 1))
        x12 = self.out(x11)

        # x13 = F.pad(x12, (x.shape[-1] - x12.shape[-1], 0))
        x13 = x12.squeeze()
        return x13

    def training_step(self, batch, batch_nb):
        x = torch.clone(batch[0])
        x = x.to(self.device)
        predictvals = torch.rand(x.shape) < 0.2
        predictvals = predictvals.to(self.device)

        inputnans = x != x
        predictvals = predictvals & ~inputnans
        x2 = torch.clone(x)
        x[predictvals] = 0.
        x_hat = self(x)

        loss = F.mse_loss(x_hat[predictvals], x2[predictvals])
        return loss

    def predict_ds(self, ds):
        """
        Given a dataset, return its predicted VOD
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        batch_ts_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_ts_hat = self(batch[0].to(self.device))
            batch_ts_hat_list.append(batch_ts_hat.cpu().detach().numpy())
        ts_hats = np.concatenate(batch_ts_hat_list) #* ds.vod_std + ds.vod_mean

        return ts_hats


class BaseGapFiller(BaseModel):
    def __init__(self, dataset, losssubset="mixed", splitglobal=True,  **kwargs):
        super(BaseGapFiller, self).__init__(dataset,  **kwargs)
        self.losssubset = losssubset
        tslen = dataset.data.shape[-1]
        self.splitseed = torch.rand(tslen)# * torch.ones([batch_size, tslen])
        self.splitglobal = splitglobal

    def training_step(self, batch, batch_nb):
        x = torch.clone(batch[0])
        x = x.to(self.device)

        if self.splitglobal:
            splitseed = self.splitseed
        else:
            splitseed = torch.rand(x.shape)
        splitseed = splitseed.to(self.device)
        predictvals = splitseed < 0.2

        inputnans = x != x
        predictvals = predictvals & ~inputnans
        x2 = torch.clone(x)
        x[predictvals] = self.nan_fillvalue
        x_hat = self(x)

        if self.losssubset=="onlygaps":
            loss = F.mse_loss(x_hat[predictvals], x2[predictvals])
        elif self.losssubset=="mixed":
            addedtrainvals = (splitseed > 0.2) & (splitseed < 0.4) & ~inputnans
            mixidx = addedtrainvals | predictvals
            loss = F.mse_loss(x_hat[mixidx], x2[mixidx])
        elif self.losssubset=="all":
            loss = F.mse_loss(x_hat[~inputnans], x2[~inputnans])
        return loss

    def predict_ds(self, ds, suffix=""):
        """
        Given a dataset, return its predicted VOD
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        batch_ts_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_ts_hat = self(batch[0].to(self.device))
            batch_ts_hat_list.append(batch_ts_hat.cpu().detach().numpy())
        ts_hats = np.concatenate(batch_ts_hat_list)# * ds.vod_std + ds.vod_mean

        return ts_hats

class LSTMGapFiller(BaseGapFiller):
    def __init__(self, dataset,**kwargs):
        super(LSTMGapFiller, self).__init__(dataset, **kwargs)
        self.lstm_1 = nn.LSTM(input_size=1, hidden_size=4, batch_first=True, bidirectional=True, num_layers=3, dropout=0.5)
        self.out = nn.Conv1d(8, 1, 1)


    def forward(self, x):
        x[x != x] = self.nan_fillvalue
        x1, _ = self.lstm_1(x[:, :, None])
        x2 = self.out(x1.transpose(1,2))

        x3 = x2.squeeze()
        return x3


class NeighbourLSTMGapFiller(BaseGapFiller):
    def __init__(self, dataset, **kwargs):
        super(NeighbourLSTMGapFiller, self).__init__(dataset,  **kwargs)
        # self.cov1 = nn.Conv1d(9, 1, 1)
        # self.gateconv = nn.Conv1d(dataset.n_neighs+1, dataset.n_neighs+1, 16)
        self.lstm_1 = nn.LSTM(input_size=dataset.n_neighs+1, hidden_size=16, batch_first=True, bidirectional=True, num_layers=3, dropout=0.5)
        self.out = nn.Conv1d(32, 1, 1)


    def forward(self, x):
        x[x != x] = self.nan_fillvalue
        # x1 = self.cov1(x)
        # x1, _ = self.lstm_1(x1)
        # x1 = self.gateconv(x)
        # F.relu(x1, True)
        x1, _ = self.lstm_1(x.transpose(1, 2))
        x2 = self.out(x1.transpose(1,2))

        x3 = x2.squeeze()
        return x3


    def training_step(self, batch, batch_nb):
        x = torch.clone(batch[0])
        x = x.to(self.device)

        if self.splitglobal:
            splitseed = self.splitseed
        else:
            splitseed = torch.rand(x.shape)
        splitseed = splitseed.to(self.device)
        predictvals = splitseed < 0.2
        centerts = torch.clone(x[:,0,:])
        inputnans = centerts != centerts
        predictvals = predictvals & ~inputnans
        x[(predictvals[:, None, :] * torch.ones_like(x)) == 1] = self.nan_fillvalue
        x_hat = self(x)

        if self.losssubset=="onlygaps":
            loss = F.mse_loss(x_hat[predictvals], centerts[predictvals])
        elif self.losssubset=="mixed":
            addedtrainvals = (splitseed > 0.2) & (splitseed < 0.4) & ~inputnans
            mixidx = addedtrainvals | predictvals
            loss = F.mse_loss(x_hat[mixidx], centerts[mixidx])
        elif self.losssubset=="all":
            loss = F.mse_loss(x_hat[~inputnans], centerts[~inputnans])
        return loss


def kl_divergence(z, mu, std, lossidx):
    #https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl[~lossidx] = np.nan
    kl = kl.nansum(-1)

    return kl


def gaussian_likelihood(x_hat, logscale, x):
    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))

def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2


def elbo(y_pred, y, mu, log_var, lossidx=None):
    # likelihood of observing y given Variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)

    # prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))

    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)

    # by taking the mean we approximate the expectation
    elbo = (likelihood + log_prior - log_p_q)
    elbo[~lossidx] = np.nan
    return elbo.nanmean()

def det_loss(y_pred, y, mu, log_var, lossidx=None):
    return -elbo(y_pred, y, mu, log_var, lossidx)


class VariationalLSTMGapFiller(BaseGapFiller):
    def __init__(self, ds, **kwargs):
        super(VariationalLSTMGapFiller, self).__init__(ds, **kwargs)
        self.lstm_1 = nn.LSTM(input_size=1, hidden_size=16, batch_first=True, bidirectional=True, num_layers=3, dropout=0.5)
        self.out = nn.Conv1d(32, 2, 1)
        self.p = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x[x != x] = 0.
        x1, _ = self.lstm_1(x[:, :, None])
        x2 = self.out(x1.transpose(1,2))
        mu = x2[:, 0]
        log_var = torch.exp(x2[:, 1])
        # std = torch.exp(log_var / 2)

        sigma = torch.exp(0.5 * log_var) + 1e-5

        eps = torch.randn_like(sigma)

        # q = torch.distributions.Normal(mu, std)
        return mu + sigma * eps, mu, log_var


    def training_step(self, batch, batch_nb):
        x = torch.clone(batch[0])
        x = x.to(self.device)
        splitseed = torch.rand(x.shape)
        splitseed = splitseed.to(self.device)
        predictvals = splitseed < 0.2

        inputnans = x != x
        predictvals = predictvals & ~inputnans
        if self.losssubset=="onlygaps":
            lossidx = predictvals
        elif self.losssubset=="mixed":
            addedtrainvals = (splitseed > 0.2) & (splitseed < 0.4) & ~inputnans
            lossidx = addedtrainvals | predictvals
        elif self.losssubset=="all":
            lossidx = ~inputnans


        x2 = torch.clone(x)
        x[predictvals] = 0.
        y_pred, mu, log_var = self(x)
        loss = det_loss(y_pred, x2, mu, log_var, lossidx)

        # z = q.rsample()

        # log_pxz = x_hat.log_prob(x2)


        # recon_loss = q.log_prob(x2)
        # recon_loss[~lossidx] = np.nan
        # recon_loss = recon_loss.nansum(1)
        # kl = kl_divergence(z, q.mean, q.stddev, lossidx)
        # elbo = kl - recon_loss
        return -loss


class BaseConvGapFiller(BaseModel):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=256, lr=0.001):
        super(BaseConvGapFiller, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
        feature_factor=4
        self.max_pool_2 = nn.MaxPool1d(2, 2)
        self.e_conv_1 = double_conv(1, 16*feature_factor)
        # self.e_batchnorm1 = nn.BatchNorm1d(16)

        self.e_conv_2 = double_conv(16*feature_factor, 32*feature_factor)
        # self.e_batchnorm2 = nn.BatchNorm1d(32)

        self.e_conv_3 = double_conv(32*feature_factor, 64*feature_factor)
        # self.e_batchnorm3 = nn.BatchNorm1d(64)

        self.d_trans_1 = nn.ConvTranspose1d(64*feature_factor, 32*feature_factor, 2, 2)
        self.d_conv_1 = double_conv(64*feature_factor, 32*feature_factor)
        # self.e_batchnorm4 = nn.BatchNorm1d(32)

        self.d_trans_2 = nn.ConvTranspose1d(32*feature_factor, 16*feature_factor, 2, 2)
        self.d_conv_2 = double_conv(32*feature_factor, 16*feature_factor)
        # self.e_batchnorm5 = nn.BatchNorm1d(16)


        self.out = nn.Conv1d(16*feature_factor, 1, 1)

    def forward(self, x):
        x[x != x] = 0.
        x1 = self.e_conv_1(x[:, None])
        x2 = self.max_pool_2(x1)
        # x2 = self.e_batchnorm1(x2)

        x3 = self.e_conv_2(x2)
        x4 = self.max_pool_2(x3)
        # x4 = self.e_batchnorm2(x4)

        x5 = self.e_conv_3(x4)
        # x5 = self.e_batchnorm3(x5)

        x6 = self.d_trans_1(x5)
        x7 = crop_tensor(x3, x6)
        x8 = self.d_conv_1(torch.cat([x6, x7], 1))
        # x8 = self.e_batchnorm4(x8)

        x9 = self.d_trans_2(x8)
        x10 = crop_tensor(x1, x9)
        x11 = self.d_conv_2(torch.cat([x9, x10], 1))
        # x11 = self.e_batchnorm5(x11)

        x12 = self.out(x11)

        x13 = F.pad(x12, (x.shape[-1] - x12.shape[-1], 0))
        x13 = x13.squeeze()
        return x13


    def training_step(self, batch, batch_nb):
        x = torch.clone(batch[0])
        x = x.to(self.device)
        predictvals = torch.rand(x.shape) < 0.2
        predictvals = predictvals.to(self.device)

        inputnans = x != x
        predictvals = predictvals & ~inputnans
        x2 = torch.clone(x)
        x[predictvals] = 0.
        x_hat = self(x)

        loss = F.mse_loss(x_hat[predictvals], x2[predictvals])
        return loss

    def predict_ds(self, ds, suffix=""):
        """
        Given a dataset, return its predicted VOD
        :param ds: torch.utils.data.Dataset
            dataset
        :return:
        """
        batch_ts_hat_list = []
        for i, batch in enumerate(DataLoader(ds, batch_size=self.batch_size, num_workers=1)):
            batch_ts_hat = self(batch[0].to(self.device))
            batch_ts_hat_list.append(batch_ts_hat.cpu().detach().numpy())
        ts_hats = np.concatenate(batch_ts_hat_list)# * ds.vod_std + ds.vod_mean

        return ts_hats





class BaseConvGapFillerAE(BaseConvGapFiller):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=256, lr=0.001):
        super(BaseConvGapFiller, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
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

    def forward(self, x):
        x[x != x] = 0.
        e2 = self.e_conv1(x[:, None])
        e3 = F.elu(e2)
        e4 = self.e_batchnorm1(e3)
        e5 = self.e_conv2(e4)
        e6 = F.elu(e5)
        e7 = self.e_batchnorm2(e6)
        e8 = e7.view(-1, e7.shape[1] * 243)
        e9 = self.e_linear1(e8)
        e10 = F.elu(e9)
        e11 = self.e_linear2(e10)
        encoding = torch.sigmoid(e11)
        d1 = self.d_linear2(encoding)
        d2 = F.elu(d1)
        d3 = self.d_linear1(d2)
        d4 = F.elu(d3)
        d5 = d4.view(-1, self.d_deconv2.in_channels, 243)
        d6 = self.d_deconv2(d5)
        d7 = self.d_batchnorm2(d6)
        d8 = F.elu(d7)
        d9 = self.d_deconv1(d8)
        d10 = d9.squeeze()
        return d10



class BaseConvGapFillerUAE(BaseConvGapFiller):
    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=256, lr=0.001):
        super(BaseConvGapFiller, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
        conv1_width = 3
        conv2_width = 4
        conv1_nfeatures = 16
        # encoder layers
        self.e_conv1 = nn.Conv1d(1, conv1_nfeatures, conv1_width, stride=2, padding_mode='circular')
        self.e_batchnorm1 = nn.BatchNorm1d(conv1_nfeatures)
        self.e_conv2 = nn.Conv1d(conv1_nfeatures, conv1_nfeatures * 2, kernel_size=conv2_width, stride=3 )
        self.e_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures * 2)
        self.e_linear1 = nn.Linear(conv1_nfeatures * 2 * 243, 64)
        self.e_linear2 = nn.Linear(64, encoding_dim)
        # decoder layers
        self.d_linear2 = nn.Linear(encoding_dim, 64)
        self.d_linear1 = nn.Linear(64, conv1_nfeatures * 2 * 243)
        self.d_conv1 = nn.Conv1d(conv1_nfeatures*4, conv1_nfeatures * 2, kernel_size=3, padding=1)

        self.d_deconv2 = nn.ConvTranspose1d(conv1_nfeatures * 2, conv1_nfeatures, kernel_size=conv2_width, stride=3)
        self.d_conv2 = nn.Conv1d(conv1_nfeatures*2, conv1_nfeatures, kernel_size=3, padding=1)

        self.d_batchnorm2 = nn.BatchNorm1d(conv1_nfeatures)
        self.d_deconv1 = nn.ConvTranspose1d(conv1_nfeatures, 1, conv1_width, stride=2, padding_mode='zeros')


        # x8 = self.d_conv_1(torch.cat([x6, x7], 1))
        # x8 = self.e_batchnorm4(x8)

        # x9 = self.d_trans_2(x8)
    def forward(self, x):
        x[x != x] = 0.
        e2 = self.e_conv1(x[:, None])
        e3 = F.elu(e2)
        e4 = self.e_batchnorm1(e3)
        e5 = self.e_conv2(e4)
        e6 = F.elu(e5)
        e7 = self.e_batchnorm2(e6)
        e8 = e7.view(x.shape[0], -1)
        e9 = self.e_linear1(e8)
        e10 = F.elu(e9)
        e11 = self.e_linear2(e10)
        encoding = torch.sigmoid(e11)
        d1 = self.d_linear2(encoding)
        d2 = F.elu(d1)
        d3 = self.d_linear1(d2)
        d4 = F.elu(d3)
        d5 = d4.view(x.shape[0], self.d_deconv2.in_channels, -1)

        d6 = self.d_conv1(torch.cat([e7, d5], 1))

        d6 = self.d_deconv2(d6)
        d7 = self.d_batchnorm2(d6)
        d8 = F.elu(d7)
        # d8 = F.pad(d8, (1,1))
        d9 = self.d_conv2(torch.cat([e4, d8], 1))

        d9 = self.d_deconv1(d9)
        d10 = d9.squeeze()
        return d10

class BaseConvAutoencoder(BaseModel):
    """
    Convolutional autoencoder. Similar to https://arxiv.org/abs/2002.03624v1
    """

    def __init__(self, dataset, encoding_dim=4, train_size_frac=0.7, batch_size=512, lr=0.001):
        super(BaseConvAutoencoder, self).__init__(dataset, encoding_dim, train_size_frac, batch_size, lr)
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

    def encoder(self, ts):
        h = self.e_conv1(ts)
        h = F.elu(h)
        h = self.e_batchnorm1(h)
        h = self.e_conv2(h)
        h = F.elu(h)
        h = self.e_batchnorm2(h)
        h = h.view(-1, h.shape[1] * 243)
        h = self.e_linear1(h)
        h = F.elu(h)
        h = self.e_linear2(h)
        h = F.sigmoid(h)
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
        loss = reconstruction_loss + t_loss + p_loss
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.training_step(batch, batch_nb)
        self.log('val_loss', loss)

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

    def encoder(self, ts):
        h = F.dropout(ts, p=0.2)
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
        h = F.sigmoid(h)

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