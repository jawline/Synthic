import math
from torch import nn
from parameters import ACTIVATION


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_dim, dropout):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            *[
                nn.Linear(dim, expansion_dim),
                ACTIVATION(),
                nn.Dropout(dropout),
                nn.Linear(expansion_dim, dim),
            ]
        )

    def forward(self, x):
        x = self.layer(x)
        return x
