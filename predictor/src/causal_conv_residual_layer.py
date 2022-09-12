import math
from torch import nn
from residual_block import ResBlock
from causal_convolution import CausalConv1d
from pointwise import Pointwise


"""
This layer combines a causal convolution layer and a residual layer together, optionally
batch normalizing the output. A stack of these blocks forms our CausalConv model.
"""


def CausalConvModelLayer(
    dim, hfactor, kernel_size, batch_norm, dilation, layer_dropout
):
    causal = CausalConv1d(dim, dim, kernel_size, dilation)
    pointwise = Pointwise(dim, hfactor)
    return ModelLayer(dim, causal, pointwise, batch_norm, layer_dropout)
