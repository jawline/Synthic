import math

import torch
from torch import nn
from local_attention import LocalAttention

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
            window_size=512,
            causal=True,
            look_backward=3,
            look_forward=0,
            dropout=0.1,
            autopad=True,
            exact_windowsize=False,
        )

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x = self.attn(q, k, v)
        return x


class PermutedAttentionBlock(nn.Module):
    def __init__(self, dim):
        super(PermutedAttentionBlock, self).__init__()
        self.attn = AttentionBlock(dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attn(x)
        x = x.permute(0, 2, 1)
        return x
