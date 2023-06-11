"""
Small script to stack the era5 images and temporally downsample the data
"""

import xarray as xr
from pathlib import Path
import os
import pandas as pd
# from dask.distributed import Client

# c = Client(n_workers=1, threads_per_worker=1)

if __name__ == "__main__":

    in_path = '/data/RADAR/Datapool_processed/ERA5/datasets/era5_regridded_imgs/'
    out_path = '/data/USERS/lmoesing/vod_encoder/data/era5_monthly.nc'
    fnames = list(Path(in_path).rglob('*.nc'))
    fnames.sort()
    fnames = fnames
    # def preprocess(ds):
    #     da = ds[['vod']]
    #     return da
    ds = xr.open_mfdataset(fnames, concat_dim='time', combine='nested'
                           # , preprocess=preprocess
                           )
    ds = ds.sortby('time')
    ds.load()
    def process_da(da):
        da_north = da[:, da['lat'] >= 0]
        da_south = da[:, da['lat'] < 0]
        da_south['time'] = da_south['time'] + pd.to_timedelta('26w')

        da_north = da_north.resample(time='1M').mean()
        da_south = da_south.resample(time='1M').mean()

        da_recombined = xr.concat([da_north, da_south], 'lat')
        return da_recombined
    das = []
    for variable in ds.keys():
        da = ds[variable]
        da_recombined = process_da(da)
        das.append(da_recombined)

    # ds = ds.resample(time='1Y').mean()
    # dims = list(ds.dims.keys())
    ds = xr.merge(das)
    # ds = ds.transpose(*dims)

    ds.attrs['temp_mean'] = ds['stl1'].mean().values
    ds.attrs['temp_std'] = ds['stl1'].std().values
    ds.attrs['prec_mean'] = ds['tp'].mean().values
    ds.attrs['prec_std'] = ds['tp'].std().values

    try:
        os.makedirs(os.path.dirname(out_path))
    except FileExistsError:
        pass

    ds.to_netcdf(out_path)


