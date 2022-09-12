import math
from torch import nn


"""
Causal convolutions prevent the convolution calculation for x[i] considering data from
any input[j] where j > i by padding the input with [kernel_size] - 1 zeros.

For example, while a standard convolution with kernel size 5 on an input [1, 2, 3, 4, 5]
may look like:
    x[0] = kernel([0, 0, 1, 2, 3])
    x[1] = kernel([0, 1, 2, 3, 4])
    x[2] = kernel([1, 2, 3, 4, 5])
    x[3] = kernel([2, 3, 4, 5, 0])
    x[3] = kernel([3, 4, 5, 0, 0])
a causal convolution padding on the same input would look like:
    x[0] = kernel([0, 0, 0, 0, 1])
    x[1] = kernel([0, 0, 0, 1, 2])
    x[2] = kernel([0, 0, 1, 2, 3])
    x[3] = kernel([0, 1, 2, 3, 4])
    x[4] = kernel([1, 2, 3, 4, 5])

If the convolution has a dilation > 1 then we will multiply the amount of padding by [dilation]
because dilated convolutions consider every [dilation * kernel_size] inputs around a target output.
"""


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, **kwargs
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)
