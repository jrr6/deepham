import torch

from torch import nn
from torch_geometric.utils.subgraph import k_hop_subgraph

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
        indices, _, _, _, = k_hop_subgraph(int(vertex), 1, edge_index)
        #FIXME: this duplicates main.py
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = torch.ones(logits.shape[0]).to(device) * float('-inf')
        mask = mask.scatter(0, indices, 0)
        mask[vertex] = float('-inf')
        mask = mask.reshape(-1, 1)

        return mask