import math

from sample import MAX_WINDOW_SIZE, BYTES_PER_ENTRY
import torch
from torch import nn


"""
Positional encoding steps encode information about where we are in a sequence of data into the data
allowing a model to change how it responds to an input value based on where it occurs in a sequence
of inputs.
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_WINDOW_SIZE * BYTES_PER_ENTRY):
        super().__init__()
        assert MAX_WINDOW_SIZE <= max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Permute the input from (batch_sz, seq_len, dim) to (seq_len, batch_sz, dim)
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]

        # Return the input to it's original shape
        x = x.permute(1, 0, 2)
        return x
