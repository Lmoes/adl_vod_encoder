import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans
import torch
from eofs.xarray import Eof
import os
import sys, getopt


def run(encoding_fname, n_clusters, tol=1e-4):
    out_dir = os.path.dirname(encoding_fname)
    encodings = xr.open_dataset(encoding_fname)['encoding']
    nanidx = np.all(np.isnan(encodings), 0)

    cluster_ids_x, _ = kmeans(torch.from_numpy(encodings.values[:, ~nanidx].T), n_clusters, tol=tol)
    cluster_ids_x = cluster_ids_x.detach().numpy()
    ds = encodings.to_dataset()
    ds["cluster_ids"] = (("lat", "lon"), np.ones(nanidx.shape))
    ds["cluster_ids"][:] = np.nan
    ds["cluster_ids"].values[~nanidx] = cluster_ids_x
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(os.path.join(out_dir, "clustered_encodings_{}_classes.nc".format(n_clusters)), encoding=encoding)

    solver = Eof(encodings.rename({'latent_variable': 'time'}))
    eofs = solver.eofs(neofs=3)
    cluster_id_eofs_mean = eofs.groupby(ds["cluster_ids"]).mean()
    cluster_id_eofs_mean_standardized = (cluster_id_eofs_mean - cluster_id_eofs_mean.quantile(0.05, 'cluster_ids')) / (
                cluster_id_eofs_mean.quantile(0.95, 'cluster_ids') - cluster_id_eofs_mean.quantile(0.05, 'cluster_ids'))
    cluster_id_eofs_mean_standardized.values[cluster_id_eofs_mean_standardized.values > 1.] = 1.
    cluster_id_eofs_mean_standardized.values[cluster_id_eofs_mean_standardized.values < 0.] = 0.

    plt.figure(figsize=(10, 4))
    ds["cluster_ids"].plot(levels=range(len(cluster_id_eofs_mean['cluster_ids']) + 1),
                    colors=[tuple(cluster_id_eofs_mean_standardized.values.T[x]) for x in
                            range(len(cluster_id_eofs_mean_standardized.T))])
    plt.ylim([-60, 80])
    plt.title("Generated Climate Classes")
    plt.savefig(os.path.join(out_dir, "{}_climate_classes.png".format(n_clusters)), dpi=300,
                bbox_inches='tight')

    eofs_standardized = (eofs - eofs.quantile(0.05, ('lon', 'lat'))) / (
            eofs.quantile(0.95, ('lon', 'lat')) - eofs.quantile(0.05, ('lon', 'lat')))
    plt.figure(figsize=(10, 4))
    eofs_standardized.plot.imshow()
    plt.ylim([-60, 80])
    plt.title("RGB of first 3 PCs")
    plt.savefig(os.path.join(out_dir, "pcs_of_encodings.png".format(n_clusters)), dpi=300,
                bbox_inches='tight')


def main(argv):
    encoding_fname = argv[0]
    n_clusters = int(argv[1])

    if len(argv) == 3:
        tol = float(argv[2])
    else:
        tol = 1e-4

    run(encoding_fname, n_clusters, tol)

if __name__ == "__main__":
    # temp_resolution = 'weekly'
    # model_name = 'DeepConvTempPrecAutoencoder_4_1'
    # encoding_fname = '/data/USERS/lmoesing/vod_encoder/output/output_{}_{}.nc'.format(temp_resolution, model_name)
    # encoding_fname = "/data/USERS/lmoesing/vod_encoder/output/encodings.nc"


    # run(encoding_fname, 10)

    main(sys.argv[1:])