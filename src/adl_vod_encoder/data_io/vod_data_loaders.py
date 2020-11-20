"""
Just some basic setup to getmyself familiar with pytorch
Build after:
https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0

"""

import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from scipy.constants.constants import zero_Celsius
import os


class VodDataset(Dataset):
    """
    VOD dataset
    """
    def __init__(self, in_path, nonans=False):
        self.da = xr.open_dataarray(in_path)
        self.da = self.da[(self.da['time.year'] >= 1989) & (self.da['time.year'] < 2017)]
        if nonans:
            self.tslocs = ~self.da.isnull().any('time')
        else:
            self.tslocs = ~self.da.isnull().all('time')

        self.vod_mean = self.da.vod_mean
        self.vod_std = self.da.vod_std
        self.data = (self.da.values[:, self.tslocs].T.astype(np.float32) - self.vod_mean) / self.vod_std
        self.sample_dim = self.data.shape[1]
        self.out_da_list = []
        self.attrs = {}
        self.add_ts(self.data*self.vod_std + self.vod_mean, 'vod_orig')

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return self.data.shape[0]

    def add_encodings(self, encodings):
        encoding_dim = encodings.shape[1]
        coords = {'latent_variable': np.arange(encoding_dim), **{c: self.da.coords[c] for c in ['lat', 'lon']}}
        da = xr.DataArray(np.nan, coords, ['latent_variable', 'lat', 'lon'], 'encoding')
        da.values[:, self.tslocs] = encodings.T
        self.out_da_list.append(da)

    def add_image(self, data, varname):
        coords = {c: self.da.coords[c] for c in ['lat', 'lon']}
        da = xr.DataArray(np.nan, coords, ['lat', 'lon'], varname)
        da.values[self.tslocs] = data
        self.out_da_list.append(da)

    def add_images(self, images):
        for image in images:
            self.add_image(images[image], image)

    def add_ts(self, data, tsname):
        da = xr.DataArray(np.nan, self.da.coords, ['time', 'lat', 'lon'], tsname)
        da.values[:, self.tslocs] = data.T
        self.out_da_list.append(da)

    def add_predictions(self, predictions):
        self.add_ts(predictions, 'vod_reconstructed')

    def add_attrs(self, attrs):
        self.attrs.update(attrs)

    def flush(self, fname):
        try:
            os.makedirs(os.path.dirname(fname))
        except FileExistsError:
            pass
        ds = xr.merge(self.out_da_list)
        ds.attrs = self.attrs
        ds.to_netcdf(fname)


class VodTempPrecDataset(VodDataset):
    """
    VOD dataset
    """
    def __init__(self, in_path, temprecipath, nonans=False):
        super(VodTempPrecDataset, self).__init__(in_path, nonans)
        self.tpds = xr.open_dataset(temprecipath)
        self.temp_mean = self.tpds.temp_mean
        self.temp_std = self.tpds.temp_std
        self.prec_mean = self.tpds.prec_mean
        self.prec_std = self.tpds.prec_std

        self.tpds = self.tpds.mean('time')
        self.tempdata = (self.tpds['stl1'].values[self.tslocs].T - self.temp_mean).astype(np.float32) / self.temp_std
        self.precdata = (self.tpds['tp'].values[self.tslocs].T - self.prec_mean).astype(np.float32) / self.prec_std

        self.add_image(self.tempdata * self.temp_std + self.temp_mean, 'temp_orig')
        self.add_image(self.precdata* self.prec_std + self.prec_mean, 'prec_orig')

    def __getitem__(self, index):
        return self.data[index], self.tempdata[index], self.precdata[index]

    def add_predictions(self, predictions):
        self.add_ts(predictions[0] * self.vod_std + self.vod_mean, 'vod_reconstructed')
        self.add_image(predictions[1].flatten() * self.temp_std + self.temp_mean, 't_hat')
        self.add_image(predictions[2].flatten() * self.prec_std + self.prec_mean, 'p_hat')
