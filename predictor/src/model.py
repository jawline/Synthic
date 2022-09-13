import math

# Sample constants we need to size the model
from sample import MAX_WINDOW_SIZE, BYTES_PER_ENTRY

# Torch stuff
import torch
import torch.nn.functional as F
from torch import nn, optim

# Local attention models go crazy and start outputting NaN if the LR is too high early for the first few epochs, but
# the LR can be increased after some initial weights are adjusted. We use this gradual warmup scheduler
# to automate that process.
from adaptive_warmup import Scheduler as AdaptiveWarmup
from positional_encoding import PositionalEncoding
from attention_block import AttentionBlock
from feed_forward import FeedForward
from residual_block import ResidualBlock
from permute import Permute


class ModelLayer(nn.Module):
    def __init__(self, dim, causal, forward, layer_dropout):
        super(ModelLayer, self).__init__()

        residual = ResidualBlock(nn.Sequential(*[causal, forward]))
        layers = [residual]

        layers.append(nn.LayerNorm(dim))

        if layer_dropout is not None:
            layers.append(nn.Dropout(p=layer_dropout))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


"""
This layer combines an attention block and a residual layer together, optionally
batch normalizing the output.
"""


def AttentionModelLayer(dim, layer_dropout):
    causal = AttentionBlock(dim)
    forward = FeedForward(dim, 1024, layer_dropout)
    return ModelLayer(dim, causal, forward, layer_dropout)


"""
A sequence to sequence model formed from a series of stacked residual blocks
that each apply a local attention layer and then a feedforward layer.
"""


class GameboyNet(nn.Module):
    def __init__(
        self,
        dim=256,
        num_attention_layers=5,
        layer_dropout=0,
    ):
        super(GameboyNet, self).__init__()

        def make_layer(i):
            return AttentionModelLayer(
                dim=dim,
                layer_dropout=layer_dropout,
            )

        self.dim = dim

        # First we embed and then add positional encodings to our input
        # TODO: Try to compress the embedding
        self.embed = nn.Embedding(dim, dim)
        self.positional_encoding = PositionalEncoding(dim)

        # Build the core of our model by stacking [layers] CausalConvModelLayer instances on top of each other.
        layers = [make_layer(layer_idx) for layer_idx in range(num_attention_layers)]
        self.layers = nn.Sequential(*layers)

        # Combine all the channels and then activate as a final step
        self.finalize = nn.Sequential(
            *[
                FeedForward(dim, dim, dropout=layer_dropout),
            ]
        )

    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.dim)
        x = self.positional_encoding(x)
        x = self.layers(x)
        x = self.finalize(x)
        return x.permute(0, 2, 1)

    """
    Calls forward and then permutes the result back to (batch_size, input_size, embedding_dim)
    from (batch_size, embedding_dim, input_size).
    """

    def predict(self, x):
        # Permute to (batch, seq_len, 256)
        return self.forward(x).permute(0, 2, 1)


"""
Stop warming up if loss starts increasing
"""


def lr_criterion(epoch, last_lr, last_loss, current_lr, current_loss):
    if epoch > 2:
        if last_loss < current_loss:
            return last_lr
        else:
            return None
    else:
        return None


"""
Load a model, either initialized with random values if [path] is None or from an existing network
saved on disk if [path] is a string.
"""


def load_model(model, path, device):

    default_lr = 0.00004

    optimizer = optim.AdamW(model.parameters(), lr=default_lr, weight_decay=0.002)

    # optimizer = optim.SGD(
    #    model.parameters(), lr=default_lr, momentum=0.9, nesterov=True
    # )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.9, min_lr=0.00000001, patience=2
    )

    # TODO: Use this
    scheduler_step = optim.lr_scheduler.StepLR(
        optimizer, step_size=999999991, gamma=0.9
    )

    model = model.to(device)

    # This needs to be after to because the optimizer decides what device to send the tensors to based on the
    # device of the model.
    if path != None:
        print("Loading from " + path)
        model.load_state_dict(torch.load(path + ".model"))
        optimizer.load_state_dict(torch.load(path + ".optimizer"))
        # scheduler = torch.load(path + ".scheduler")
    else:
        # Fresh model so start with some adaptive warmup
        scheduler = AdaptiveWarmup(
            optimizer,
            start_lr=0.00000001,
            end_lr=default_lr,
            num_steps=8,
            criterion=lr_criterion,
            underlying_scheduler=scheduler,
            pass_through_loss_to_underlying=True,
        )

    return model, optimizer, scheduler, scheduler_step


def load_gameboy_net(path, device):
    return load_model(GameboyNet(), path, device)
