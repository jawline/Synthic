import math
from torch import nn


# TODO: Try and autoencode down the complex sequence into a single action
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        # TODO: Make this a deep network
        self.input_dim = input_dim
        layer1 = nn.Linear(input_dim, 1024)
        layer2 = nn.Linear(1024, output_dim)

        self.layers = nn.Sequential(*[layer1, nn.ReLU(), layer2])

    def forward(self, x):
        # print("Start:", x.shape)
        origin_shape = x.shape
        x = x.reshape((x.shape[0], -1, self.input_dim))
        x = self.layers(x)
        x = x.reshape(origin_shape[0], -1, origin_shape[2])
        # print("Final:", x.shape)
        return x
