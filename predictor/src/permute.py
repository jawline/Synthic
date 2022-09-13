import math

import torch
from torch import nn
from local_attention import LocalAttention


class Permute(nn.Module):
    def __init__(self, layer):
        super(Permute, self).__init__()
        self.layer = layer

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer(x)
        x = x.permute(0, 2, 1)
        return x
