"""
Just some basic setup to getmyself familiar with pytorch
Build after:
https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0

"""

import xarray as xr
import numpy as np
from torch.utils.data import Dataset


class VodDataset(Dataset):
    """
    VOD dataset. Current limitations:
        - only post 1987 as it is incomplete
    """
    def __init__(self, in_path, nonans=False):
        self.da = xr.open_dataarray(in_path)
        self.da = self.da[(self.da['time.year'] > 1989) & self.da['time.year'] < 2017]
        if nonans:
            self.tslocs = ~self.da.isnull().any('time')
        else:
            self.tslocs = ~self.da.isnull().all('time')
        self.data = self.da.values[:, self.tslocs].T.astype(np.float32)
        self.time = self.da['time']
        self.sample_dim = self.data.shape[1]
        # self.out_da_list = []

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return self.data.shape[0]

    # def add_encodings(self, encodings):
    #
    #     encoding_dim = encodings.shape[1]
    #     coords = {'latent_variable': np.arange(encoding_dim), **{c: self.da.coords[c] for c in ['lat', 'lon']}}
    #     da = xr.DataArray(np.nan, coords, ['latent_variable', 'lat', 'lon'], 'encoding')
    #     da.values[:, self.tslocs] = encodings.T
    #     self.out_da_list.append(da)
    #
    # def add_clusteridx(self, cluster_idx):
    #     coords = {c: self.da.coords[c] for c in ['lat', 'lon']}
    #     da = xr.DataArray(np.nan, coords, ['lat', 'lon'], 'cluster_idx')
    #     da.values[self.tslocs] = cluster_idx
    #     self.out_da_list.append(da)
    #
    # def flush_outputs(self, fname):
    #     try:
    #         os.makedirs(os.path.dirname(fname))
    #     except FileExistsError:
    #         pass
    #     ds = xr.merge(self.out_da_list)
    #     ds.to_netcdf(fname)

