import torch
import torch_geometric

from torch import nn


class Mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(logits, edges, vertex):
        """
        @param logits: logits for all vertices

        @param graph: Data encoding of a graph

        @param vertex: current vertex to mask

        @returns: a tensor of logits masked to change all non-adjacent vertices to negative infinity
        """

        # compute neighbors for the inputted vertex using the edges
        indices, _, _, _, = torch_geometric.util.k_hop_subgraph(vertex, 1, edges)
        mask = torch.ones(logits.shape[0]) * float('-inf')
        mask.scatter(0, indices, 0)

        return mask


class MLP(nn.Module):
    def __init__(self, embed_size, hidden_layer_size, relu_alpha):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(embed_size, hidden_layer_size),
            nn.LeakyReLU(negative_slope=relu_alpha),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(negative_slope=relu_alpha),
            nn.Linear(hidden_layer_size, 1),
            # TODO Mask?(),
            # Mask()
        )
        self.mask = Mask()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor, edges, vertex):
        x = self.seq(x)
        x = x + Mask.forward(x, edges, vertex)
        return self.softmax(x)
