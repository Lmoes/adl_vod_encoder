import os
import xarray as xr
import matplotlib.pyplot as plt
from torch import rand, reshape, save, load, from_numpy
import numpy as np
class Plotter(object):

    def __init__(self, path_in, metrics_path):
        self.path_in = path_in
        self.ds = xr.open_dataset(path_in)

        self.model_name = "_".join(os.path.basename(self.path_in)[:-3].split('_')[2:])
        self.path_out = os.path.join(os.path.dirname(os.path.dirname(self.path_in)), "figures", self.model_name)

        self.metrics = load(metrics_path).metrics

        try:
            os.makedirs(self.path_out)
        except FileExistsError:
            pass

    def plot_all(self):
        self.plot_imgs()
        self.plot_diff_imgs()
        self.plot_metrics()
    def plot_imgs(self):
        self.plot_img("vod_orig")
        self.plot_img("vod_reconstructed")
        self.plot_img("vod_td_1.0")
        self.plot_img("vod_td_2.0")

    def plot_diff_imgs(self):
        self.plot_diff_img("vod_reconstructed", "vod_orig")
        self.plot_diff_img("vod_td_1.0", "vod_reconstructed")
        self.plot_diff_img("vod_td_2.0", "vod_reconstructed")

        try:
            self.plot_diff_img("t_hat", "temp_orig")
            self.plot_diff_img("t_hat_1.0", "t_hat")
            self.plot_diff_img("t_hat_2.0", "t_hat")
        except KeyError:
            pass

        try:
            self.plot_diff_img("p_hat", "prec_orig")
            self.plot_diff_img("p_hat_1.0", "p_hat")
            self.plot_diff_img("p_hat_2.0", "p_hat")
        except KeyError:
            pass

    def plot_metrics(self):
        self.plot_metric("val_r2")
        self.plot_metric("val_loss")

    def _get_metric(self, metric):
        try:
            return np.array([x[metric].numpy() for x in self.metrics if metric in x])
        except TypeError:
            return np.array([x[metric].cpu().numpy() for x in self.metrics if metric in x])
    def plot_metric(self, metric):
        y = self._get_metric(metric)
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(1, len(y)+1), y)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.savefig(
            os.path.join(self.path_out, metric + ".png"),
            dpi=300,
            bbox_inches='tight')
        pass
    def plot_ts(self, vars, lat, lon):
        raise NotImplementedError

    def plot_img(self, var, time=None):
        if time is None:
            da = self.ds[var]
            m = da.mean("time")
            plt.figure(figsize=(10, 4))
            m.plot(robust=True)
            plt.ylim([-60, 80])
            name = "Mean_{}".format(var)
            plt.title(name)
            plt.savefig(
                os.path.join(self.path_out, name + ".png"),
                dpi=300,
                bbox_inches='tight')
        else:
            raise NotImplementedError

    def plot_diff_img(self, var_a, var_b, time=None, relative=False):
        if not relative:
            self.plot_diff_img(var_a, var_b, time, relative=True)
        if time is None:
            da_a = self.ds[var_a]
            da_b = self.ds[var_b]
            if relative:
                m = da_a / da_b
                name = "Mean_diff_{}_div_{}".format(var_a, var_b)
            else:
                m = da_a - da_b
                name = "Mean_diff_{}_minus_{}".format(var_a, var_b)

            try:
                m = m.mean("time")
            except ValueError:
                pass
            plt.figure(figsize=(10, 4))
            m.plot(robust=True)
            plt.ylim([-60, 80])
            plt.title(name)
            plt.savefig(
                os.path.join(self.path_out, name + ".png"),
                dpi=300,
                bbox_inches='tight')
        else:
            raise NotImplementedError