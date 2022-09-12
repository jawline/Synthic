import math
from torch import nn

"""
Use a 1D convolution of size 1 to visit each data point in the input
independently.

TODO: Document this module better.
"""


class Pointwise(nn.Module):
    def __init__(self, dim, hfactor):
        super(Pointwise, self).__init__()

        expanded_dim = dim * hfactor

        layers = [
            nn.Conv1d(dim, expanded_dim, 1),
            nn.ReLU(),
            nn.Conv1d(expanded_dim, dim, 1),
        ]

        self.transform = nn.Sequential(*layers)

    def forward(self, x):
        return self.transform(x)
