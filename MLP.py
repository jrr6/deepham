import torch

from torch import nn

EMBED_SIZE = 512
HIDDEN_LAYER_SIZE = 256
RELU_ALPHA = 0.1


class Mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(graph):
        """
        @param graph:
        """
        pass


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(EMBED_SIZE, HIDDEN_LAYER_SIZE),
            nn.LeakyReLU(negative_slope=RELU_ALPHA),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.LeakyReLU(negative_slope=RELU_ALPHA),
            nn.Linear(HIDDEN_LAYER_SIZE, 1),
            # TODO Mask?(),
            nn.Softmax()
        )

    def forward(self, x: nn.Tensor):
        return self.seq(x)
