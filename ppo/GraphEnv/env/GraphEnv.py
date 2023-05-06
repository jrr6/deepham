import gym
import numpy as np
import torch

from gym import spaces
from torch_geometric.data import Data
from data import generate_semirandom_hampath_graph



class GraphEnv(gym.Env):
    def __init__(
        self, graph: Data | None = None, starting_vertex_index: int | None = None
    ):
        super(GraphEnv, self).__init__()

        if graph is None:
            self.initial_graph, self.initial_starting_vertex = generate_semirandom_hampath_graph(30, 0, 15, 10)
        else:
            self.initial_graph, self.initial_starting_vertex = graph, starting_vertex_index

        self.num_vertices = self.initial_graph.x.size()[0]


        self.action_space = spaces.Discrete(self.num_vertices)
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0, high=1, shape=(self.num_vertices, self.num_vertices)),
            "edge_index": spaces.Box(low=0, high=self.num_vertices, shape=(2,)),
            "current_vertex": spaces.Discrete(self.num_vertices)
        })

    def reset(self, new_graph=False):
        if new_graph:
            self.graph, self.current_vertex = generate_semirandom_hampath_graph(30, 0, 15, 10)
        else:
            self.graph, self.current_vertex = self.initial_graph.clone(), self.initial_starting_vertex

        self.path = [self.current_vertex]

        observation, info = self._get_obs(), self._get_info()
        return observation, info

    def step(self, action: int):
        # Check if it is valid to move from current_vertex to action
        edge = torch.Tensor([[action, self.current_vertex]]).T

        if not (torch.all(np.equal(self.graph.edge_index, edge), dim=0).any().item()):  # type: ignore
            raise RuntimeError(
                f"Tried to step to non-adjacent vertex {action} from vertex {self.current_vertex}")

        # Remove all the edges from current_vertex in the graph so we don't revisit it
        not_adjacent_to_old_vertex = torch.all(self.graph.edge_index != self.current_vertex, dim=0)
        self.graph = Data(x=self.graph.x, edge_index=self.graph.edge_index[:, not_adjacent_to_old_vertex])

        # Update the current_vertex
        self.current_vertex = action
        self.path.append(self.current_vertex)

        is_curr_vertex_isolated: bool = torch.all(self.graph.edge_index != self.curr_vertex_index).item()  # type: ignore

        observation, info = self._get_obs(), self._get_info()
        reward = len(self.path)
        terminated = len(self.path) == self.num_vertices or is_curr_vertex_isolated

        return observation, reward, terminated, info

    def _get_obs(self):
        return {"x": self.graph.x, "edge_index": self.graph.edge_index, "current_vertex": self.current_vertex}

    def _get_info(self):
        return {}