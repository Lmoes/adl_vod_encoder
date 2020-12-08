"""
Small script to stack the era5 images and temporally downsample the data
"""

import xarray as xr
from pathlib import Path
import os
import pandas as pd


if __name__ == "__main__":

    in_path = '/data-read/RADAR/Datapool_processed/ERA5/datasets/era5_regridded_imgs/'
    out_path = '/data-write/USERS/lmoesing/vod_encoder/data/era5mean.nc'
    fnames = list(Path(in_path).rglob('*.nc'))
    fnames.sort()
    fnames = fnames
    def preprocess(ds):
        da = ds['vod']
        return da
    ds = xr.open_mfdataset(fnames, concat_dim='time', combine='nested'
                           # , preprocess=preprocess
                           )

    ds = ds.resample(time='1Y').mean()

    ds.attrs['temp_mean'] = ds['stl1'].mean().values
    ds.attrs['temp_std'] = ds['stl1'].std().values
    ds.attrs['prec_mean'] = ds['tp'].mean().values
    ds.attrs['prec_std'] = ds['tp'].std().values

    try:
        os.makedirs(os.path.dirname(out_path))
    except FileExistsError:
        pass

    ds.to_netcdf(out_path)


