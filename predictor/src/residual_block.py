import math
from torch import nn


"""
A residual block trains a layer to predict the residual of a previous
output (i.e, how much do we need to nudge the output by the get the
correct result).
"""


class ResidualBlock(nn.Module):
    def __init__(self, layer):
        super(ResidualBlock, self).__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)
