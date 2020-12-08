===============
adl_vod_encoder
===============


This project is about automatically extracting features from Vegetation Optical Depth (VOD) time series.
These features are then clustered using a shallow learner (currently k-means) to generate global vegetation clusters.


VOD Preprocessing
===========
https://github.com/Lmoesinger/adl_vod_encoder/blob/main/src/adl_vod_encoder/preprocessing/vodca_preprocessing.py

The original data are daily global images with a quarter degree resolution (1440 x 720 pixels). They range from 1987-08 to 2017-06, but only 1989-01-01 to 2016-12-31 is used as having fully years makes things easier and 1988 has some issues. Values on the southern hemisphere are also shifted by 6 months, so that their seasons align with the northern winter.

The data are downsampled to weekly values and saved in a netcdf stack. There are a few reasons for downsampling:
 - The data has missing values, and by taking weekly means we reduce the number of gaps.
 - The original dataset is quite large (~300GB), downsampled (and by dropping some unnecessary columns) it is at 13.3GB.
 - The original data is quite noisy, therefore sub-weekly variations are more a result of noise rather than the climate.

Auxiliary data Preprocessing
===========
https://github.com/Lmoesinger/adl_vod_encoder/blob/main/src/adl_vod_encoder/preprocessing/era5_preprocessing.py

We also use ERA5 (a climate reanalysis dataset) precipitation and surface temperature, which are traditionally used for vegetation classifications.
We the temporal means, which are side tasks for the autoencoder to predict from the encoding.


Normalization/standardization
===========
All data are standardized before feeding it to the network to make them use the whole possible range and centered around 0.

standardized(x) = (x - mean(x)) / std(x)

Autoencoder architecture
===========
https://github.com/Lmoesinger/adl_vod_encoder/blob/0f2faf0d3a3824bb8113e0d97e76e05b7b773e14/src/adl_vod_encoder/models/autoencoders.py#L141

Currently, the setup is quite basic:

- The encoder is just one layer that has the size of the encoding dimension
- The decoder is just one layer that has the size of the input dimension

I experimented around with convolutional autodencoders, but they did not perform better. I will still try to improve this.

Additionally, the network also tries to predict the mean precipitation and mean temperature.
 This is mostly done as a regularization, since it forces the autoencoder to produce an encoding
 layer that actually contains information and does not just map every training time series to a specific encoding.
 This is currently done also just with one linear layer. In the future I will experiment with more layers, as the
 temperature prediction is quite poor currently.


Error Metrics for neural network
============
There are three losses: One for reconstructing the VOD time series, and one for predicting the temperature and precipitation each.
I use mean square error everywhere, and weight all errors equally. Therefore, currently:

loss = mse(predicted_vod, original_vod) + mse(predicted_precipitation, target_precipitation) + mse(predicted_temperature, target_temperature)
Currently the temperature and vod loss are very low, while the temperature loss is a lot higher.

Error Metrics for clustering
============
This is a bit difficult as there is no ground truth. While we could make up some metrics like spatial coherence, these cant capture whether the classification makes sense. So it makes more sense to do a qualitative analysis of the clusters. Here are some results:

[plot](deliverables/results/output_weekly_ConvTempPrecAutoencoder.png)

![Image of Yaktocat](https://storage.googleapis.com/gweb-uniblog-publish-prod/images/earth-4k.max-1000x1000.jpg)

Notes for myself
===========
pytochlightning template:
https://github.com/PyTorchLightning/deep-learning-project-template


Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
