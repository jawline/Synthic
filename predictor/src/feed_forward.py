import math
from torch import nn


class PermutedFeedForward(nn.Module):
    def __init__(self, dim, expansion_dim, dropout):
        super(PermutedFeedForward, self).__init__()
        self.layer = nn.Sequential(
            *[
                nn.Linear(dim, expansion_dim),
                nn.ReLU(),  # TODO: Play with different activations?
                nn.Dropout(dropout),
                nn.Linear(expansion_dim, dim),
            ]
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer(x)
        x = x.permute(0, 2, 1)
        return x
