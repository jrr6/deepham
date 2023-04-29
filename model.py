import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.distributions import Categorical
from functools import reduce
from MLP import MLP

VertexSet = torch.Tensor
EdgeIndices = torch.Tensor
Vertex = torch.Tensor

STATE_SIZE_TODO = -1

class DeepHamModel(nn.Module):
    def __init__(self,
                 node_embedding_size: int = 512,
                 hidden_layer_size: int = 256,
                 relu_alpha: float = 0.1):
        super(DeepHamModel, self).__init__()
        self.actor = DeepHamActor(node_embedding_size, hidden_layer_size, relu_alpha)
        self.critic = DeepHamCritic(STATE_SIZE_TODO, hidden_layer_size, 3, relu_alpha)

    def forward(self, state: tuple[VertexSet, EdgeIndices, Vertex]):
        # TODO: Proper inputs
        probs = self.actor(state)
        value = self.critic(state)

        return probs, value


class DeepHamActor(nn.Module):
    def __init__(self, node_embedding_size: int = 512, hidden_layer_size: int = 256, relu_alpha: float = 0.1):
        super(DeepHamActor, self).__init__()

        self.conv1 = GCNConv(-1, node_embedding_size)
        self.conv2 = GCNConv(node_embedding_size, node_embedding_size)
        self.conv3 = GCNConv(node_embedding_size, node_embedding_size)

        self.predictor = MLP(node_embedding_size, hidden_layer_size, relu_alpha)

    def forward(self, state):
        vertices, edge_index, current_vertex = state

        vertex_embeddings = self.conv1(vertices.float(), edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv2(vertex_embeddings, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv3(vertex_embeddings, edge_index)
        vertex_embeddings = vertices.tanh()

        return self.predictor(vertices, edge_index, current_vertex)


class DeepHamCritic(nn.Module):
    def __init__(self, input_dim, hidden_layer_size: int = 256, relu_alpha: float = 0.1):
        super(DeepHamCritic, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.layer3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output = nn.Linear(hidden_layer_size, 1)
        self.relu_alpha = relu_alpha

    def forward(self, x):
        return self.output(
            F.leaky_relu(self.layer3(
                F.leaky_relu(self.layer2(
                    F.leaky_relu(self.layer1(x), self.relu_alpha)
                ), self.relu_alpha)
            ), self.relu_alpha)
        )

# References:
# https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
# https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py


class DeepHamLoss(nn.Module):
    def forward(self, actor_out, critic_out, discounted_reward):
        advantage = discounted_reward - critic_out
        log_prob = Categorical(actor_out).log_prob()

        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.smooth_l1_loss(critic_out, torch.tensor([discounted_reward]))

        return actor_loss + critic_loss
