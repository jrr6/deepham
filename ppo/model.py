import torch

from mask import Mask
from torch import nn
from torch_geometric.nn import Sequential, GATv2Conv

class DeepHamActor(nn.Module):
    def __init__(self):
        super(DeepHamActor, self).__init__()
        self.seq = Sequential("x, edge_index, current_vertex", [
            #                           GNN                                  #
            (GATv2Conv(-1,  512),       "x, edge_index                 -> x"),
            (nn.Tanh(),                 "x                             -> x"),
            (GATv2Conv(512, 512),       "x, edge_index                 -> x"),
            (nn.Tanh(),                 "x                             -> x"),
            (GATv2Conv(512, 512),       "x, edge_index                 -> x"),
            (nn.Tanh(),                 "x                             -> x"),
            #                           MLP                                  #
            (nn.Linear(512, 512),       "x                             -> x"),
            (nn.LeakyReLU(),            "x                             -> x"),
            (nn.Linear(512, 512),       "x                             -> x"),
            (nn.LeakyReLU(),            "x                             -> x"),
            (nn.Linear(512, 512),       "x                             -> x"),
            (Mask(),                    "x, edge_index, current_vertex -> x"),
            (nn.Softmax(dim=0),         "x                             -> x"),
        ])

    def forward(self, x, edge_index, current_vertex):
        return self.seq(x, edge_index, current_vertex).reshape((x.size()[0],))

class DeepHamCritic(nn.Module):
    def __init__(self):
        super(DeepHamCritic, self).__init__()
        self.seq = Sequential("x, edge_index", [
            # GNN to compute node embeddings
            (GATv2Conv(-1,  512),       "x, edge_index -> x"),
            (torch.nn.Tanh(),           "x             -> x"),
            (GATv2Conv(512, 512),       "x, edge_index -> x"),
            (torch.nn.Tanh(),           "x             -> x"),
            (GATv2Conv(512, 512),       "x, edge_index -> x"),
            (torch.nn.Tanh(),           "x             -> x"),
            # MLP to compute value approximation
            (torch.nn.Linear(512, 512), "x             -> x"),
            (torch.nn.LeakyReLU(),      "x             -> x"),
            (torch.nn.Linear(512, 512), "x             -> x"),
            (torch.nn.LeakyReLU(),      "x             -> x"),
            (torch.nn.Linear(512, 1),   "x             -> x"),
        ])

    def forward(self, x, edge_index):
        return self.seq(x, edge_index)

class SuperDeepHam(nn.Module):
    def __init__(self):
        super(SuperDeepHam, self).__init__()
        self.seq = Sequential("x, edge_index", [
            #                           GNN                                  #
            (GATv2Conv(-1,  512),       "x, edge_index                 -> x"),
            (nn.Tanh(),                 "x                             -> x"),
            (GATv2Conv(512, 512),       "x, edge_index                 -> x"),
            (nn.Tanh(),                 "x                             -> x"),
            (GATv2Conv(512, 512),       "x, edge_index                 -> x"),
            (nn.Tanh(),                 "x                             -> x"),
            #                           MLP                                  #
            (nn.Linear(512, 512),       "x                             -> x"),
            (nn.LeakyReLU(),            "x                             -> x"),
            (nn.Linear(512, 512),       "x                             -> x"),
            (nn.LeakyReLU(),            "x                             -> x"),
            (nn.Linear(512, 512),       "x                             -> x"),
            (nn.Softmax(dim=0),         "x                             -> x")
        ])

    def forward(self, x, edge_index):
        return self.seq(x, edge_index)