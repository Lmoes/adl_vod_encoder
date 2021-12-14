"""
Just some basic setup to getmyself familiar with pytorch
Build after:
https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0

"""

import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import os
from torch.utils.data import random_split


class VodDataset(Dataset):
    """
    VOD dataset loader, also handles writing of encoding
    """
    def __init__(self, in_path, nonans=False, equalyearsize=False, split="all", anoms=True, neighbours=0):
        self.da = xr.open_dataarray(in_path)
        self.vod_mean = self.da.vod_mean
        self.vod_std = self.da.vod_std
        self.out_da_list = []
        self.attrs = {}
        self.anoms = anoms
        self.neighbours = neighbours

        self.da = self.da.sel(time=slice("1989-01-01", "2016"))
        self.da.load()
        if anoms:
            da_smoothed = self.da.rolling({"time": 4}, center=True, min_periods=1).mean()
            woy = da_smoothed["time"].dt.weekofyear
            self.clim = da_smoothed.groupby(woy).apply(lambda x: x.mean("time"))
            anoms = self.da.groupby(woy).apply(lambda x: x - self.clim.sel(weekofyear=x.time.dt.weekofyear.values[0])).drop("weekofyear")
            self.da = anoms
            self.vod_mean = 0.
            self.vod_std = self.da.std("time")

        self.da = (self.da - self.vod_mean) / self.vod_std

        np.random.seed(42)
        train_test = np.random.rand(self.da.shape[0])
        train_test = train_test[:, None, None] * np.ones(self.da.shape)
        np.random.seed(None)
        self.trainidx = train_test < 0.8

        if equalyearsize:
            self.da = xr.concat([self.da[self.da['time.year'] == year][:52] for year in np.unique(self.da['time.year'])], 'time')

        if nonans:
            self.tslocs = ~self.da.isnull().any('time')
        else:
            self.tslocs = (~self.da.isnull()).sum('time') > 50


        self.changefilter("all")
        self.add_ts(self.data, 'vod_orig')
        self.changefilter("test")
        self.add_ts(self.data, 'vod_orig_test')

        self.changefilter(split)
        self.sample_dim = self.data.shape[1]

        ## making an index for each GPI which the neighbour GPIs are
        self.n_neighs = (neighbours*2+1)**2 - 1
        if self.neighbours != 0:
            locnums = np.zeros_like(self.tslocs, float)
            locnums[:] = np.nan
            nlocs = self.tslocs.sum()
            locnums[self.tslocs] = np.arange(nlocs)
            n_list = []
            for row in range(-neighbours, neighbours + 1):
                for col in range(-neighbours, neighbours + 1):
                    if (col == 0) and (row == 0):
                        continue
                    else:
                        n_list.append(np.pad(locnums, [[neighbours + row, neighbours - row],
                                                       [neighbours + col, neighbours - col]], "wrap"))

            locnighs = np.stack(n_list)
            locnighs = locnighs[:,neighbours:-neighbours, neighbours:-neighbours]
            locnighs = locnighs.reshape((self.n_neighs, -1)).T
            self.locnighs = locnighs[self.tslocs.values.ravel()]




    def changefilter(self, split="all"):
        """
        Changes the dataset between training/test/all
        :param split:
        :return:
        """
        self.data = self.da.values[:, self.tslocs].T.astype(np.float32)
        if split == "train":
            self.data[~self.trainidx[:, self.tslocs].T] = np.nan
        if split == "test":
            self.data[self.trainidx[:, self.tslocs].T] = np.nan
        self.split = split

    def anomstoraw(self, anoms):
        woy = anoms["time"].dt.weekofyear
        raw = anoms.groupby(woy).apply(lambda x: x + self.clim.sel(weekofyear=x.time.dt.weekofyear.values[0])).drop("weekofyear")
        raw.name = anoms.name
        return raw


    def __getitem__(self, index):
        if self.neighbours == 0:
            return (self.data[index], )
        else:
            # return (self.data[list(range(9))].T, )

            neigidx = self.locnighs[index]
            tmpl = np.empty([self.n_neighs + 1, self.sample_dim]).astype(np.float32)
            tmpl[:] = np.nan
            neighisnan = neigidx != neigidx
            dataidx = neigidx[~neighisnan].astype(int)
            data = self.data[np.concatenate([[index], dataidx])]
            tmpl[~np.concatenate([[False], neighisnan]), :] = data
            # dummydata = np.ones((self.n_neighs + 1 - data.shape[0], data.shape[1])).astype(np.float32)
            # dummydata[:] = np.nan
            return (tmpl, )

    def __len__(self):
        return self.data.shape[0]

    def add_encodings(self, encodings):
        """
        add the encodings to the output
        :param encodings: np.array
            an array containing the encoding
        :return:
        """
        encoding_dim = encodings.shape[-1]
        if encodings.ndim == 2:
            coords = {'latent_variable': np.arange(encoding_dim), **{c: self.da.coords[c] for c in ['lat', 'lon']}}
            da = xr.DataArray(np.nan, coords, ['latent_variable', 'lat', 'lon'], 'encoding')
            da.values[:, self.tslocs] = encodings.T

        else:
            years = np.unique(self.da.coords['time.year'])
            coords = {'latent_variable': np.arange(encoding_dim),
                      'year': years,
                      **{c: self.da.coords[c] for c in ['lat', 'lon']}}
            da = xr.DataArray(np.nan, coords, ['latent_variable', 'year', 'lat', 'lon'], 'encoding')
            da.values[:, :, self.tslocs] = encodings.T

        self.out_da_list.append(da)

    def add_image(self, data, varname):
        """
        add ad am image to the output
        :param data: np.array
            the image data
        :param varname: string
            name of the variable
        :return:
        """
        varname = varname + "_" + self.split
        if data.ndim == 1:
            coords = {c: self.da.coords[c] for c in ['lat', 'lon']}
            da = xr.DataArray(np.nan, coords, ['lat', 'lon'], varname)
            da.values[self.tslocs] = data
        else:
            years = np.unique(self.da.coords['time.year'])
            coords = {'year': years, **{c: self.da.coords[c] for c in ['lat', 'lon']}}
            da = xr.DataArray(np.nan, coords, ['year', 'lat', 'lon'], varname)
            da.values[:, self.tslocs] = data.T

        self.out_da_list.append(da)

    def add_images(self, images):
        """
        add a dict of images to the output
        :param images: dict {name: np.arrray}
            dict of images to be added
        :return:
        """
        for image in images:
            self.add_image(images[image], image)

    def add_ts(self, data, tsname):
        """
        add a time series to the output
        :param data: np.array
            the time series
        :param tsname: string
            name of the variable
        :return:
        """
        da = xr.DataArray(np.nan, self.da.coords, ['time', 'lat', 'lon'], tsname + "_" + self.split)
        da.values[:, self.tslocs] = data.T
        da = (da * self.vod_std + self.vod_mean).rename(da.name)

        if self.anoms:
            self.out_da_list.append(da.rename(da.name + "_anoms"))
            da = self.anomstoraw(da)
        self.out_da_list.append(da)

    def add_predictions(self, predictions):
        """
        Add the predictions to the output
        :param predictions: np.array
            predicted vod values
        :return:
        """
        self.add_ts(predictions, 'vod_reconstructed')

    def add_attrs(self, attrs):
        """
        Update the attributes of the netcdf
        :param attrs: dict
            attribites to be added
        :return:
        """
        self.attrs.update(attrs)

    def flush(self, fname):
        """
        write all outputs to disk
        :param fname: filename of output file
        :return:
        """
        try:
            os.makedirs(os.path.dirname(fname))
        except FileExistsError:
            pass
        ds = xr.merge(self.out_da_list)
        ds.attrs = self.attrs
        comp = dict(zlib=True, complevel=5)

        try:
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(fname, encoding=encoding)
        except AttributeError:
            encoding = {ds.name: comp}
            ds.to_netcdf(fname, encoding=encoding)


class SMDataSet(VodDataset):


    def __init__(self, in_path, nonans=False, split="all", neighbours=0):
        """
        :param in_path: filename of sm data
        :param nonans: if true, loads only ts with no nans in it. if false, loads all ts with at least 50 observations.
        :param split: Whether to load "all" data or only "test" or "train"(ing) data
        :param neighbours: number of neighbours

        # neighbours is an integer indicating how many neighbouring ts should be used
    #     0: Just center ts (use LSTMGapFiller)
    #     1: 3-neighbourhood = 9 ts total (use NeighbourLSTMGapFiller)
    #     2: 5-neighbourhood = 25 total (use NeighbourLSTMGapFiller)
    #     3: 7-neibourhood = 49 total (use NeighbourLSTMGapFiller)
    #     4: etc....
        """
        ds = xr.open_dataset(in_path)
        self.da = ds["sm_anom"]
        self.vod_mean = self.da.mean("time")
        self.vod_std = self.da.std("time")
        self.out_da_list = []
        self.attrs = {}
        self.neighbours = neighbours
        self.anoms=True

        self.da = self.da.sel(time=slice("1989-01-01", "2016"))
        self.da.load()

        self.clim = self.read_clim(in_path)

        self.da = (self.da - self.vod_mean) / self.vod_std

        np.random.seed(42)
        train_test = np.random.rand(self.da.shape[0])
        train_test = train_test[:, None, None] * np.ones(self.da.shape)
        np.random.seed(None)
        self.trainidx = train_test < 0.8

        if nonans:
            self.tslocs = ~self.da.isnull().any('time')
        else:
            self.tslocs = (~self.da.isnull()).sum('time') > 50

        self.changefilter("all")
        self.add_ts(self.data, 'vod_orig')
        self.changefilter("test")
        self.add_ts(self.data, 'vod_orig_test')

        self.changefilter(split)
        self.sample_dim = self.data.shape[1]
        self.n_neighs = (neighbours*2+1)**2 - 1
        if self.neighbours != 0:
            # locnums = np.arange(self.tslocs.size).reshape(self.tslocs.shape)
            locnums = np.zeros_like(self.tslocs, float)
            locnums[:] = np.nan
            nlocs = self.tslocs.sum()
            locnums[self.tslocs] = np.arange(nlocs)
            # locnums = np.arange(16).reshape((4,4))
            n_list = []
            for row in range(-neighbours, neighbours + 1):
                for col in range(-neighbours, neighbours + 1):
                    if (col == 0) and (row == 0):
                        continue
                    else:
                        # n_list.append(np.pad(locnums, [[1+row, 1-row], [1+col, 1-col]], "wrap"))
                        n_list.append(np.pad(locnums, [[neighbours + row, neighbours - row],
                                                       [neighbours + col, neighbours - col]], "wrap"))

            locnighs = np.stack(n_list)
            locnighs = locnighs[:,neighbours:-neighbours, neighbours:-neighbours]
            locnighs = locnighs.reshape((self.n_neighs, -1)).T
            self.locnighs = locnighs[self.tslocs.values.ravel()]

    def read_clim(self, in_path):
        smdir = os.path.join(os.path.dirname(in_path), "sm_clim/")
        fnames = [os.path.join(smdir, x) for x in os.listdir(smdir) if x.endswith(".nc")]
        ds_clim = xr.open_mfdataset(fnames)
        da_clim = ds_clim["sm"]
        da_clim = da_clim.transpose("time", "lat", "lon")
        da_clim = da_clim.groupby(da_clim.time.dt.month).mean()
        return da_clim

    def anomstoraw(self, anoms):
        woy = anoms["time"].dt.month
        raw = anoms.groupby(woy).apply(lambda x: x + self.clim.sel(month=x.time.dt.month.values[0])).drop("month")
        raw.name = anoms.name
        return raw



class VodNeighbourDataset(VodDataset):
    def __init__(self, in_path, nonans=False, equalyearsize=False, split="all", anoms=True):
        self.da = xr.open_dataarray(in_path)
        self.vod_mean = self.da.vod_mean
        self.vod_std = self.da.vod_std
        self.out_da_list = []
        self.attrs = {}
        self.anoms = anoms

        self.da = self.da.sel(time=slice("1989-01-01", "2016"))
        self.da.load()
        if anoms:
            da_smoothed = self.da.rolling({"time": 4}, center=True, min_periods=1).mean()
            woy = da_smoothed["time"].dt.weekofyear
            self.clim = da_smoothed.groupby(woy).apply(lambda x: x.mean("time"))
            anoms = self.da.groupby(woy).apply(lambda x: x - self.clim.sel(week=x.time.dt.weekofyear.values[0])).drop("week")
            self.da = anoms
            self.vod_mean = 0.
            self.vod_std = self.da.std("time")

        self.da = (self.da - self.vod_mean) / self.vod_std

        np.random.seed(42)
        train_test = np.random.rand(self.da.shape[0])
        train_test = train_test[:, None, None] * np.ones(self.da.shape)
        np.random.seed(None)
        self.trainidx = train_test < 0.8

        if equalyearsize:
            self.da = xr.concat([self.da[self.da['time.year'] == year][:52] for year in np.unique(self.da['time.year'])], 'time')

        if nonans:
            self.tslocs = ~self.da.isnull().any('time')
        else:
            self.tslocs = (~self.da.isnull()).sum('time') > 50


class VodTempPrecDataset(VodDataset):
    """
    Dataloader/writer that loads VOD, precipitation and temperature data
    """
    def __init__(self, in_path, temprecipath, nonans=False, equalyearsize=False):
        super(VodTempPrecDataset, self).__init__(in_path, nonans, equalyearsize)
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
        """
        add the vod/temp/prec preduictions to the output
        :param predictions: tuple of np.arrays
            (vod_predictions, temperature_predictions, precipitation_predictions), each a np.array
        :return:
        """
        self.add_ts(predictions[0] * self.vod_std + self.vod_mean, 'vod_reconstructed')
        self.add_image(predictions[1].flatten() * self.temp_std + self.temp_mean, 't_hat')
        self.add_image(predictions[2].flatten() * self.prec_std + self.prec_mean, 'p_hat')
