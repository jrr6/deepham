import GraphEnv
import torch

from mask import Mask
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch_geometric.nn import Sequential, GATv2Conv

# PyTorch Parameters
device = "cpu" if torch.has_cuda else "cuda:0"

# Model Hyperparameters
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1

SUB_BATCH_SIZE = 64
NUM_EPOCHS = 10
PPO_CLIP_FACTOR = 0.2
DISCOUNT_FACTOR = 0.99
LAMBDA = 0.95
ENTROPY_EPSILON = 1e-4


# Step (1): Define an Environment
kwargs = {"num_vertices": 10, "num_edges": 20, "delta_e": 15, "regenerate_graph": True}
env = GymEnv("GraphEnv/GraphEnv-v0", **kwargs)

print(f"input_spec: {env.input_spec}")
print(f"action_spec: {env.action_spec}")
print(f"observation_spec: {env.observation_spec}")
print(f"reward_spec: {env.reward_spec}")

# Step (2): Initialize Actor (Policy) and Critic (Value) Networks

actor_net = Sequential("x, edge_index, current_vertex", [
    #                           GNN                                  #
    (GATv2Conv(-1,  512),       "x, edge_index                 -> x"),
    (torch.nn.Tanh(),           "x                             -> x"),
    (GATv2Conv(512, 512),       "x, edge_index                 -> x"),
    (torch.nn.Tanh(),           "x                             -> x"),
    (GATv2Conv(512, 512),       "x, edge_index                 -> x"),
    (torch.nn.Tanh(),           "x                             -> x"),
    #                           MLP                                  #
    (torch.nn.Linear(512, 512), "x                             -> x"),
    (torch.nn.LeakyReLU(),      "x                             -> x"),
    (torch.nn.Linear(512, 512), "x                             -> x"),
    (torch.nn.LeakyReLU(),      "x                             -> x"),
    (torch.nn.Linear(512, 512), "x                             -> x"),
    (Mask(),                    "x, edge_index, current_vertex -> x"),
    (torch.nn.Softmax(dim=0),   "x                             -> x")
])

critic_net = Sequential("x, edge_index, current_vertex", [
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


policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["probs"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action.spec,
    in_keys=["probs"],
    distribution_class=Categorical,
    distribution_kwargs=...,
    return_log_prob=True #?
)

critic_net = ...
value_module = ValueOperator(
    module=critic_net,
    in_keys=["observation"]
)

print(f"Running Policy: {policy_module(env.reset())}")
print(f"Running Value: {value_module(env.reset())}")

advantage_module = GAE(
    gamma=DISCOUNT_FACTOR, lmabda=LAMBDA, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    advantage_key="advantage",
    clip_epsilon=PPO_CLIP_FACTOR,
    entropy_bonus=bool(ENTROPY_EPSILON),
    entropy_coef=ENTROPY_EPSILON,
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), LEARNING_RATE)