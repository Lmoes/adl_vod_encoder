# -*- coding: utf-8 -*-

import unittest
from src.adl_vod_encoder.data_io.vod_data_loaders import VodDataset, VodTempPrecDataset
import os
import numpy as np


class TestDatasets(unittest.TestCase):
    """
    Test whether the dataloaders work as expected
    """
    def setUp(self):
        testdatadir = os.path.join(os.getcwd(), 'testdata')
        VodDatasetpath = os.path.join(testdatadir, 'v01_erafrozen_k_weeklysample.nc')
        era5Datasetpath = os.path.join(testdatadir, 'era5meansample.nc')
        self.vod_ds = VodDataset(VodDatasetpath)
        self.era5_ds = VodTempPrecDataset(VodDatasetpath, era5Datasetpath)

    def test_vod_equality(self):
        """
        Test whether both loaders load the same vod data.
        :return:
        """
        dataA = self.vod_ds.data[~np.isnan(self.vod_ds.data)]
        dataB = self.era5_ds.data[~np.isnan(self.era5_ds.data)]
        self.assertEqual(dataA.tolist(), dataB.tolist(), 'test datasets are not read equally')

    """
    Check if datasets have mean==0 and std==1 after standardization.
    """
    def test_std_vod(self):
        self.assertAlmostEqual(np.nanstd(self.vod_ds.data), 1, delta=0.2, msg='standardized vod values have a variance != 1')

    def test_std_precipitation(self):
        self.assertAlmostEqual(np.nanstd(self.era5_ds.precdata), 1, delta=0.2, msg='standardized prec values have a variance != 1')

    def test_std_temperature(self):
        self.assertAlmostEqual(np.nanstd(self.era5_ds.tempdata), 1, delta=0.2, msg='standardized temp values have a variance != 1')

    def test_mean_vod(self):
        self.assertAlmostEqual(np.nanmean(self.vod_ds.data), 0, delta=0.2, msg='standardized vod values have a mean != 0')

    def test_mean_precipitation(self):
        self.assertAlmostEqual(np.nanmean(self.era5_ds.precdata), 0, delta=0.2, msg='standardized prec values have a mean != 0')

    def test_mean_temperature(self):
        self.assertAlmostEqual(np.nanmean(self.era5_ds.tempdata), 0, delta=0.2, msg='standardized temp values have a mean != 0')


if __name__ == '__main__':
    unittest.main()
