import torch_geometric as pyg

# might need some of these
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

# might not


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
import GraphEnv
import gym

env = gym.make("GraphEnv/GraphEnv-v0")