from __future__ import annotations

import torch
import torch_geometric as pyg
import numpy as np

from torch_geometric.data import Data
from data import generate_hampath_graph


Reward = int
IsDone = bool
IsTruncated = bool


class GraphState:
    def __init__(self, graph: None | Data = None):
        self.reset(graph)

    def reset(self, graph: None | Data = None) -> GraphState:
        self.graph = graph if graph is not None else generate_hampath_graph(3, 10)
        self.num_vertices = self.graph.x.size()[0]
        self.path = []

        self.curr_vertex = torch.zeros(size=(self.num_vertices,))
        self.curr_vertex_index = -1

        return self

    def step(self, action_idx: int) -> tuple[GraphState, Reward, IsDone, IsTruncated]:
        # Check if it is valid to move from current_vertex to action
        edge = torch.Tensor([[action_idx, self.curr_vertex_index]]).T

        if not (self.curr_vertex_index == -1 or
                torch.all(np.equal(self.graph.edge_index, edge), dim=0).any().item()):  # type: ignore
            raise RuntimeError("Tried to step to a non-adjacent vertex")

        # Remove all the edges from current_vertex in the graph so we don't revisit it
        not_adjacent_to_old_vertex = torch.all(self.graph.edge_index != self.curr_vertex_index, dim=0)
        self.graph = Data(x=self.graph.x, edge_index=self.graph.edge_index[:, not_adjacent_to_old_vertex])

        # Update the current_vertex
        self.curr_vertex_index = action_idx
        self.curr_vertex = torch.eye(self.num_vertices)[action_idx, :]
        self.path.append(self.curr_vertex_index)

        is_curr_vertex_isolated: bool = torch.all(
            self.graph.edge_index != self.curr_vertex_index).item()  # type: ignore

        reward = len(self.path)  # TODO: Implement Reward
        isDone = len(self.path) == self.num_vertices or is_curr_vertex_isolated
        isTruncated = isDone and len(self.path) != self.num_vertices

        return self, reward, isDone, isTruncated