import os
import xarray as xr
import numpy as np


class OutputWriter(object):

    def __init__(self, ds):
        self.coords = ds.da.coords
        self.tslocs = ds.tslocs
        self.out_da_list = []

    def add_encodings(self, encodings):
        encoding_dim = encodings.shape[1]
        coords = {'latent_variable': np.arange(encoding_dim), **{c: self.coords[c] for c in ['lat', 'lon']}}
        da = xr.DataArray(np.nan, coords, ['latent_variable', 'lat', 'lon'], 'encoding')
        da.values[:, self.tslocs] = encodings.T
        self.out_da_list.append(da)

    def add_clusteridx(self, cluster_idx):
        coords = {c: self.coords[c] for c in ['lat', 'lon']}
        da = xr.DataArray(np.nan, coords, ['lat', 'lon'], 'cluster_idx')
        da.values[self.tslocs] = cluster_idx
        self.out_da_list.append(da)

    def add_ts(self, data, tsname):
        da = xr.DataArray(np.nan, self.coords, ['time', 'lat', 'lon'], tsname)
        da.values[:, self.tslocs] = data.T
        self.out_da_list.append(da)
        pass

    def flush(self, fname):
        try:
            os.makedirs(os.path.dirname(fname))
        except FileExistsError:
            pass
        ds = xr.merge(self.out_da_list)
        ds.to_netcdf(fname)
