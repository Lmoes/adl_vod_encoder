import xarray as xr
from eofs.xarray import Eof
import matplotlib.pyplot as plt
import os


def plot_clusters(encoding_fname):
    """
    Just a little function to plot the encodings.
    :param encoding_fname: string, "path/filename.nc" of the encoding
    :return:
    """
    ds = xr.open_dataset(encoding_fname)
    encoding_da = ds['encoding']
    cluster_da = ds['cluster_ids']
    solver = Eof(encoding_da.rename({'latent_variable': 'time'}))
    eofs = solver.eofs(neofs=3)
    cluster_id_eofs_mean = eofs.groupby(cluster_da).mean()
    cluster_id_eofs_mean_standardized = (cluster_id_eofs_mean - cluster_id_eofs_mean.quantile(0.05, 'cluster_ids')) / (
                cluster_id_eofs_mean.quantile(0.95, 'cluster_ids') - cluster_id_eofs_mean.quantile(0.05, 'cluster_ids'))
    cluster_id_eofs_mean_standardized.values[cluster_id_eofs_mean_standardized.values > 1.] = 1.
    cluster_id_eofs_mean_standardized.values[cluster_id_eofs_mean_standardized.values < 0.] = 0.

    plt.figure(figsize=(10, 4))
    cluster_da.plot(levels=range(len(cluster_id_eofs_mean['cluster_ids']) + 1),
                    colors=[tuple(cluster_id_eofs_mean_standardized.values.T[x]) for x in
                            range(len(cluster_id_eofs_mean_standardized.T))])
    plt.ylim([-60, 80])
    plt.title(os.path.basename(encoding_fname).split('_')[2])
    plt.savefig(os.path.join(os.path.dirname(encoding_fname), os.path.basename(encoding_fname)[:-3] + '_clusters.png'), dpi=300,
                bbox_inches='tight')

    eofs_standardized = (eofs - eofs.quantile(0.05, ('lon', 'lat'))) / (
            eofs.quantile(0.95, ('lon', 'lat')) - eofs.quantile(0.05, ('lon', 'lat')))
    plt.figure(figsize=(10, 4))
    eofs_standardized.plot.imshow()
    plt.ylim([-60, 80])
    plt.title(os.path.basename(encoding_fname).split('_')[2])
    plt.savefig(os.path.join(os.path.dirname(encoding_fname), os.path.basename(encoding_fname)[:-3] + '_3eofs.png'), dpi=300,
                bbox_inches='tight')

if __name__ == "__main__":
    temp_resolution = 'weekly'
    model_name = 'DeepConvTempPrecAutoencoder_2'
    encoding_fname = '/data/USERS/lmoesing/vod_encoder/output/output_{}_{}.nc'.format(temp_resolution, model_name)
    plot_clusters(encoding_fname)
