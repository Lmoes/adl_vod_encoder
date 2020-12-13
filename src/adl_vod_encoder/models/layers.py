"""
Classes that define custom layers for our models
"""


import pytorch_lightning as pl


class Reshape(pl.LightningModule):
    """
    pl.LightningModule that reshapes a tensor. Necessary for use in torch.nn.Sequential(...) to build the model.
    """
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class Squeeze(pl.LightningModule):
    """
    pl.LightningModule that squeezes a tensor. Necessary for use in torch.nn.Sequential(...) to build the model.
    """
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class Split(pl.LightningModule):
    """
    pl.LightningModule that splits a tensor. Necessary for use in torch.nn.Sequential(...) to build the model.
    """
    def __init__(self, split_size_or_sections, dim=0):
        super(Split, self).__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, x):
        return x.split(self.split_size_or_sections, self.dim)


class View(pl.LightningModule):

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)
