"""
Small script to stack the vodca images and temorally downsample the data
"""

import xarray as xr
from pathlib import Path
import os
import pandas as pd


if __name__ == "__main__":

    in_path = '/data-write/RADAR/vod_merged/v01_erafrozen/1.0/data/vod_K/merge_passive_vod_K_imgs'
    out_path = '/data/USERS/lmoesing/vod_encoder/data/v01_erafrozen_k_monthly2.nc'
    fnames = list(Path(in_path).rglob('*.nc'))
    fnames.sort()
    fnames = fnames
    def preprocess(ds):
        da = ds['vod']
        return da

    da = xr.open_mfdataset(fnames, concat_dim='time', combine='nested',
                           preprocess=preprocess
                           )
    da = da.sortby('time')

    # shift southern data time index by half a year so the seasons align with the northern hemisphere
    da_north = da[:, da['lat'] >= 0]
    da_south = da[:, da['lat'] < 0]
    da_south['time'] = da_south['time'] + pd.to_timedelta('26w')

    da_north = da_north.resample(time='1M').mean()
    da_south = da_south.resample(time='1M').mean()

    da_recombined = xr.concat([da_north, da_south], 'lat')
    try:
        os.makedirs(os.path.dirname(out_path))
    except FileExistsError:
        pass
    da_recombined.attrs['vod_mean'] = da_recombined.mean().values
    da_recombined.attrs['vod_std'] = da_recombined.std().values
    da_recombined.to_netcdf(out_path)


