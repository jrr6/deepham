import gym
import numpy as np
import torch

from gym import spaces
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from data import generate_semirandom_hampath_graph


class GraphEnv(gym.Env):
    def __init__(
        self,
        graph: Data | None = None,
        starting_vertex_index: int | None = None,
        num_vertices = 30,
        num_edges = 20,
        delta_e = 15,
        regenerate_graph = True
    ):
        super(GraphEnv, self).__init__()

        self.num_vertices = num_vertices
        self.num_edges = num_edges
        self.delta_e = delta_e
        self.regenerate_graph = regenerate_graph

        if graph is None:
            self.initial_graph, self.initial_starting_vertex = generate_semirandom_hampath_graph(self.num_vertices, 0, self.num_edges, self.delta_e)
        else:
            self.initial_graph, self.initial_starting_vertex = graph, starting_vertex_index

        self.edge_observation_size = self.initial_graph.edge_index.size()[1]

        self.action_space = spaces.Box(low=0, high=self.num_vertices, shape=[])
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=float("-inf"), high=float("inf"), shape=(self.num_vertices, 4)),
            "edge_index": spaces.Box(low=0, high=self.num_vertices, shape=(self.num_vertices, self.num_vertices), dtype=np.int64),
            "current_vertex": spaces.Box(low=-1, high=self.num_vertices, shape=[], dtype=np.int64)
        })

    def reset(self, seed=None, options=None):
        if self.regenerate_graph:
            self.graph, self.current_vertex = generate_semirandom_hampath_graph(self.num_vertices, 0, self.num_edges, self.delta_e)
        else:
            self.graph, self.current_vertex = self.initial_graph.clone(), self.initial_starting_vertex

        self.path = [self.current_vertex]

        self.edge_observation_size = self.initial_graph.edge_index.size()[1]

        observation, info = self._get_obs(), self._get_info()
        return observation, info

    def step(self, action: np.ndarray):
        # Check if it is valid to move from current_vertex to action
        edge = torch.Tensor([[int(action), self.current_vertex]]).T

        if not (torch.all(np.equal(self.graph.edge_index, edge), dim=0).any().item()):  # type: ignore
            raise RuntimeError(
                f"Tried to step to non-adjacent vertex {action} from vertex {self.current_vertex}")

        # Remove all the edges from current_vertex in the graph so we don't revisit it
        not_adjacent_to_old_vertex = torch.all(self.graph.edge_index != self.current_vertex, dim=0)

        self.graph = Data(x=self.graph.x, edge_index=self.graph.edge_index[:, not_adjacent_to_old_vertex])

        # Update the current_vertex
        self.current_vertex = int(action)
        self.path.append(self.current_vertex)

        print(self.current_vertex)

        is_curr_vertex_isolated: bool = torch.all(self.graph.edge_index != self.current_vertex).item()  # type: ignore

        observation, info = self._get_obs(), self._get_info()
        reward = len(self.path)
        terminated = len(self.path) == self.num_vertices or is_curr_vertex_isolated

        return observation, reward, terminated, info

    def _get_obs(self):
        return {
            "x": self.graph.x.numpy(),
            "edge_index": to_dense_adj(self.graph.edge_index, max_num_nodes=self.num_vertices),
            "current_vertex": self.current_vertex
        }

    def _get_info(self):
        return {}