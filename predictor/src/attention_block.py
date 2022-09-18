import math

import torch
from torch import nn
from local_attention import LocalAttention

from parameters import (
    ATTENTION_WINDOW_SIZE,
    ATTENTION_DROPOUT,
    ATTENTION_WINDOW_LOOKBACK,
)

"""
TODO: Add documentation on why we should use attention blocks.
"""


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn = LocalAttention(
            dim=dim,
            window_size=ATTENTION_WINDOW_SIZE,
            causal=True,
            look_backward=ATTENTION_WINDOW_LOOKBACK,
            look_forward=0,
            dropout=ATTENTION_DROPOUT,
            autopad=True,
            exact_windowsize=False,
        )

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x = self.attn(q, k, v)
        return x
