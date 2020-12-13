import xarray as xr
from sklearn.decomposition import PCA
from eofs.xarray import Eof
import matplotlib.pyplot as plt
import os


def plot_clusters(encoding_fname):
    ds = xr.open_dataset(encoding_fname)
    encoding_da = ds['encoding']
    cluster_da = ds['cluster_ids']
    solver = Eof(encoding_da.rename({'latent_variable': 'time'}))
    eofs = solver.eofs(neofs=3)
    cluster_id_eofs_mean = eofs.groupby(cluster_da).mean()
    cluster_id_eofs_mean_standardized = (cluster_id_eofs_mean - cluster_id_eofs_mean.min('cluster_ids')) / (
                cluster_id_eofs_mean.max('cluster_ids') - cluster_id_eofs_mean.min('cluster_ids'))

    plt.figure(figsize=(10, 4))
    cluster_da.plot(levels=range(11),
                    colors=[tuple(cluster_id_eofs_mean_standardized.values.T[x]) for x in
                            range(len(cluster_id_eofs_mean_standardized.T))])
    plt.ylim([-60, 80])
    plt.title(os.path.basename(encoding_fname.split('_')[-1])[:-3])
    plt.savefig(os.path.join(os.path.dirname(encoding_fname), os.path.basename(encoding_fname)[:-3] + '.png'), dpi=300,
                bbox_inches='tight')
    pass

if __name__ == "__main__":
    temp_resolution = 'weekly'
    model_name = 'ConvTempPrecAutoencoder'
    # model_name = 'BaseTempPrecAutoencoder'
    # model_name = 'BaseModel'
    encoding_fname = '/data/USERS/lmoesing/vod_encoder/output/output_{}_{}.nc'.format(temp_resolution, model_name)
    plot_clusters(encoding_fname)
