import torch

from torch_geometric.utils.subgraph import k_hop_subgraph

from torch import nn


class Mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, edge_index: torch.Tensor, vertex: int):
        """
        @param logits: logits for all vertices

        @param graph: Data encoding of a graph

        @param vertex: current vertex to mask

        @returns: a tensor of logits masked to change all non-adjacent vertices to negative infinity
        """

        # if at start, no mask
        if vertex == -1:
            return torch.zeros_like(logits)

        # compute neighbors for the inputted vertex using the edges
        indices, _, _, _, = k_hop_subgraph(vertex, 1, edge_index)
        mask = torch.ones(logits.shape[0]) * float('-inf')
        mask = mask.scatter(0, indices, 0)
        mask[vertex] = float('-inf')
        mask = mask.reshape(-1, 1)

        return mask


class MLP(nn.Module):
    def __init__(self, hidden_layer_size, relu_alpha):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LazyLinear(hidden_layer_size),
            nn.LeakyReLU(negative_slope=relu_alpha),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(negative_slope=relu_alpha),
            nn.Linear(hidden_layer_size, 1)
        )
        self.mask = Mask()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, current_vertex: int):
        x = self.seq(x)
        x = x + self.mask(x, edge_index, current_vertex)
        return self.softmax(x)
