import torch
import torch_geometric as pyg

from torch_geometric.data import Data
from data import generate_hampath_graph
from typing import NamedTuple

State = NamedTuple("State", [("graph", Data)])
Reward = int
IsDone = bool
IsTruncated = bool


class GraphState:
    def __init__(self, graph: None | Data = None):
        self.reset(graph)

    def reset(self, graph: None | Data = None) -> State:
        # TODO: Provide resonable arguments for the generate_hampath_graph function
        self.graph = graph if graph is not None else generate_hampath_graph(3, 1)

        return State(self.graph)

    def step(self, action: torch.Tensor) -> tuple[State, Reward, IsDone, IsTruncated]:
        return (State(self.graph), 0, False, False)
