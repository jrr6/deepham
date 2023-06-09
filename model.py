import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv
from GraphState import GraphState, Reward
from MLP import MLP

VertexSet = torch.Tensor
EdgeIndices = torch.Tensor
Vertex = torch.Tensor

EPSILON = np.finfo(np.float32).eps.item()  # avoid-div-by-0 factor


class DeepHamAgent(nn.Module):
    def __init__(
        self,
        node_embedding_size: int = 512,
        hidden_layer_size: int = 256,
        relu_alpha: float = 0.1,
    ):
        super(DeepHamAgent, self).__init__()
        self.actor = DeepHamActor(node_embedding_size, hidden_layer_size, relu_alpha)
        self.critic = DeepHamCritic(node_embedding_size, hidden_layer_size, relu_alpha)

    def forward(self, state: tuple[VertexSet, EdgeIndices, Vertex]):
        # TODO: Proper inputs
        probs = self.actor(state)
        value = self.critic(state)

        return probs, value


class DeepHamActor(nn.Module):
    def __init__(
        self,
        node_embedding_size: int = 512,
        hidden_layer_size: int = 256,
        relu_alpha: float = 0.1,
    ):
        super(DeepHamActor, self).__init__()

        self.conv1 = GCNConv(-1, node_embedding_size)
        self.conv2 = GCNConv(node_embedding_size, node_embedding_size)
        self.conv3 = GCNConv(node_embedding_size, node_embedding_size)

        self.predictor = MLP(hidden_layer_size, relu_alpha)

    def forward(self, state: GraphState):
        vertices: torch.Tensor = state.graph.x
        edge_index: torch.Tensor = state.graph.edge_index
        current_vertex_idx: int = state.curr_vertex_index

        vertex_embeddings: torch.Tensor = self.conv1(vertices, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv2(vertex_embeddings, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv3(vertex_embeddings, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()

        # print(vertex_embeddings)

        return self.predictor(vertex_embeddings, edge_index, current_vertex_idx)


# FIXME: this should be more like the actor? not copy/pasted?
class DeepHamCritic(nn.Module):
    def __init__(
        self,
        node_embedding_size: int = 512,
        hidden_layer_size: int = 256,
        relu_alpha: float = 0.1,
    ):
        super(DeepHamCritic, self).__init__()
        self.conv1 = GCNConv(-1, node_embedding_size)
        self.conv2 = GCNConv(node_embedding_size, node_embedding_size)
        self.conv3 = GCNConv(node_embedding_size, node_embedding_size)

        self.dense = nn.Sequential(
            nn.LazyLinear(hidden_layer_size),
            nn.LeakyReLU(relu_alpha),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(relu_alpha),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(relu_alpha),
            nn.Linear(hidden_layer_size, 1),
        )
        self.relu_alpha = relu_alpha

    def forward(self, state):
        vertices: torch.Tensor = state.graph.x
        edge_index: torch.Tensor = state.graph.edge_index

        vertex_embeddings: torch.Tensor = self.conv1(vertices, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv2(vertex_embeddings, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv3(vertex_embeddings, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()

        flattened_embeddings = torch.flatten(vertex_embeddings)
        output = self.dense(flattened_embeddings)

        # print(output)
        return output


# References:
# https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
# https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py


class DeepHamLoss(nn.Module):
    def forward(
        self,
        log_probs: list[torch.Tensor],
        values: list[torch.Tensor],
        rewards: list[Reward],
        gamma: float = 0.99,
    ) -> torch.Tensor:
        discounted_reward: float = 0.0
        discounted_rewards: list[float] = []
        for reward in rewards[::-1]:
            discounted_reward = reward + gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards.reverse()

        dr_tensor: torch.Tensor = torch.tensor(discounted_rewards)
        # dr_tensor = (dr_tensor - dr_tensor.mean()) / (dr_tensor.std() + EPSILON)

        actor_losses: list[torch.Tensor] = []
        critic_losses: list[torch.Tensor] = []

        for log_prob, value, dr in zip(log_probs, values, dr_tensor):
            advantage = dr - value
            actor_losses.append(-log_prob * advantage.detach())
            # TODO: maybe come back to this
            # critic_losses.append(F.smooth_l1_loss(value, torch.tensor([dr])))
            critic_losses.append(advantage**2)

        # print("actor loss", torch.stack(actor_losses).mean().item(), "critic loss", torch.stack(critic_losses).mean().item())

        return torch.stack(actor_losses).mean() + torch.stack(critic_losses).mean()
