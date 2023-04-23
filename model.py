import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from MLP import MLP


class DeepHam(nn.Module):
    def __init__(self, embed_size=16):
        super(DeepHam, self).__init__()

        ### Hyperparameters ###
        self.node_embedding_size = embed_size
        #######################

        self.conv1 = GCNConv()
        self.conv2 = GCNConv()
        self.conv3 = GCNConv()

        self.predictor = MLP()

    def forward(self, vertices, edge_index):
        out = []
        original_vertices = vertices.detach()  # breaks autograd connection

        # FIXME: do something more thoughtful (this breaks our loss)
        for _ in range(len(original_vertices)):

            # TODO: Call GNNs here
            vertices = self.conv1(vertices, edge_index)
            vertices = vertices.tanh()
            vertices = self.conv2(vertices, edge_index)
            vertices = vertices.tanh()
            vertices = self.conv3(vertices, edge_index)
            vertices = vertices.tanh()

            probs = self.predictor(vertices)

            chosen_vertex = vertices[torch.argmax(probs)]

            out.append(chosen_vertex)

            # TODO: delete chosen vertex (and its incident edges)

        return out

# TODO: Come up with a better name


class Dist_Loss(nn.Module):
    def __init__(self):
        super(Dist_Loss, self).__init__()

    # Computes difference in length between two tensors
    # Squared (because that's a good idea)
    def forward(self, X: nn.Tensor, Y: nn.Tensor):
        return (len(X) - len(Y)) ** 2
