import GraphEnv
import torch

from DenseToSparseTransform import DenseToSparseTransform
from model import DeepHamActor, DeepHamCritic
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# PyTorch Parameters
device = "cpu" if torch.has_cuda else "cuda"
print(f"Detected Device: {device}")


# Model Hyperparameters
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1

SUB_BATCH_SIZE = 64
NUM_EPOCHS = 10
PPO_CLIP_FACTOR = 0.2
DISCOUNT_FACTOR = 0.99
LAMBDA = 0.95
ENTROPY_EPSILON = 1e-4
#

# Step (1): Define an Environment
kwargs = {"num_vertices": 10, "num_edges": 5, "delta_e": 0, "regenerate_graph": True}
base_env = GymEnv("GraphEnv/GraphEnv-v0", **kwargs)
env = TransformedEnv(
    base_env,
    DenseToSparseTransform(in_keys=["edge_index"], out_keys=["edge_index"])
)

# Step (2): Initialize Actor (Policy) and Critic (Value) Networks

actor_net = DeepHamActor()
critic_net = DeepHamCritic()

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["probs"],
)


policy_module = ProbabilisticActor(
    module=policy_module, # type: ignore
    spec=env.action_spec,
    in_keys=["probs"],
    distribution_class=Categorical,
    distribution_kwargs=...,
    return_log_prob=True
)

value_module = ValueOperator(
    module=critic_net,
    in_keys=["observation"]
)

state = env.reset()

print(f"New Edge Index: {state['edge_index']}")


# advantage_module = GAE(
#     gamma=DISCOUNT_FACTOR, lmbda=LAMBDA, value_network=value_module, average_gae=True
# )

# loss_module = ClipPPOLoss(
#     actor=policy_module,
#     critic=value_module,
#     advantage_key="advantage",
#     clip_epsilon=PPO_CLIP_FACTOR,
#     entropy_bonus=bool(ENTROPY_EPSILON),
#     entropy_coef=ENTROPY_EPSILON,
#     value_target_key=advantage_module.value_target_key,
#     critic_coef=1.0,
#     gamma=0.99,
#     loss_critic_type="smooth_l1",
# )

# optim = torch.optim.Adam(loss_module.parameters(), LEARNING_RATE)