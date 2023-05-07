import GraphEnv # ! Do not remove this import, this is required for the custom OpenAI gym environment
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime
from DenseToSparseTransform import DenseToSparseTransform
from io import TextIOWrapper
from itertools import repeat
from IPython.display import display, clear_output
from model import DeepHamActor, DeepHamCritic
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torch_geometric.utils import k_hop_subgraph
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from typing import Callable

# PyTorch Parameters
device = "cuda" if torch.has_cuda else "cpu"
print(f"Detected Device: {torch.cuda.get_device_name(0) if torch.has_cuda else 'cpu'}")


# Model Hyperparameters
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1

SUB_BATCH_SIZE = 10
NUM_EPOCHS = 10
PPO_CLIP_FACTOR = 0.2
DISCOUNT_FACTOR = 0.99
LAMBDA = 0.95
ENTROPY_EPSILON = 1e-4

PRINT_FREQUENCY = 10
#

def run_episode(policy_module, value_module, env: TransformedEnv, optimizer: torch.optim.Optimizer, criterion: ClipPPOLoss) -> torch.Tensor:
    state = env.reset()

    rewards = []
    log_probs = []
    values = []

    done = state["done"]
    while not done:
        policy_module(state)
        value_module(state)

        state = env.step(state)["next"]

        done = state["done"]

    optimizer.zero_grad()
    # loss = criterion(...)
    # loss.backward()
    optimizer.step()

    # return loss

def run_random_episode(policy_module, value_module, env, optimizer, criterion) -> torch.Tensor:
    state = env.reset()
    done = state["done"]

    while not done:
        actions, _, _, _ = k_hop_subgraph(state["current_vertex"], 1, state["edge_index"])
        actions = actions.detach().cpu().numpy()
        actions = np.delete(actions, np.where(actions == int(state["current_vertex"])))
        action = np.random.choice(actions).item()

        state, _, done, _ = env.step(action)

    return torch.tensor(0, device=device)

def train_model(visualize=True, notebook=False, random=False, episodes=500, num_verts=30, num_edges=15, delta_e=10, prepopulate=True):
    if notebook:
        log_file = create_log_file()
        log_fn = write_log_file(log_file)
    else:
        log_fn = print

    # Step (1): Define an Environment
    kwargs = {"num_vertices": num_verts, "num_edges": num_edges, "delta_e": delta_e, "regenerate_graph": True}
    base_env = GymEnv("GraphEnv/GraphEnv-v0", **kwargs)
    transformed_env = TransformedEnv(base_env, DenseToSparseTransform(in_keys=["edge_index"], out_keys=["edge_index"]))
    ###

    # Step (2): Initialize Actor (Policy) and Critic (Value) Networks
    actor_net = DeepHamActor()
    critic_net = DeepHamCritic()

    policy_module = ProbabilisticActor(
        module = TensorDictModule(actor_net, in_keys=["x", "edge_index", "current_vertex"], out_keys=["probs"]), # type: ignore
        spec=transformed_env.action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True
    )

    value_module = ValueOperator(
        module=critic_net,
        in_keys=["x", "edge_index"]
    )
    ###

    # Step (3): Initialize the parameters of the models
    policy_module(transformed_env.reset())
    value_module(transformed_env.reset())
    ###

    # Step (4): Initialize Loss, Advantage, and Optimizer
    advantage_module = GAE(
        gamma=DISCOUNT_FACTOR, lmbda=LAMBDA, value_network=value_module, average_gae=True
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

    optimizer = torch.optim.Adam(loss_module.parameters(), LEARNING_RATE)

    lengths = []
    losses = []

    log_fn(("Starting training" if not random else "Starting random simulation")
           + f":\tverts = {num_verts}\tedges = {num_edges}\tdelta_e = {delta_e}\tprepopulate_start={prepopulate}")
    fig, [loss_ax, length_ax, graph_ax] = plt.subplots(1, 3)

    #! Currently not using ReplayBuffer
    env_iterator = repeat(transformed_env, episodes)

    for i, env in enumerate(env_iterator):
        if not random:
            loss = run_episode(policy_module, value_module, env, optimizer, loss_module)
        else:
            loss = run_random_episode(policy_module, value_module, env, optimizer, loss_module)

        losses.append(loss.detach().cpu().numpy())

        if i % PRINT_FREQUENCY == PRINT_FREQUENCY - 1:
            log_fn(f"Episode {i + 1}: path len = {len(env.path)}\t path = {env.path}")
            if visualize:
                graph_ax.clear()
                graph_ax.set_title("Graph")

                # Labels
                path_strs = list(map(str, env.path))
                line_length = 10
                lines = [", ".join(path_strs[i:i+line_length]) for i in range(0, len(env.path), line_length)]
                path_multiline = ",\n".join(lines)
                graph_ax.set_xlabel(f"[{path_multiline}]\nLength: {len(env.path)}", multialignment='left')

                # Drawing
                nx_graph = pyg.utils.to_networkx(env.initial_graph, to_undirected=True)  # type: ignore
                pos = nx.kamada_kawai_layout(nx_graph) # type: ignore
                non_path_nodes = [v for v in nx_graph.nodes if v not in env.path]
                nx.draw_networkx_nodes(nx_graph, pos, nodelist=non_path_nodes, ax=graph_ax, node_color="gray")  # type: ignore
                colors = list(range(len(env.path)))  # match the gradient to the ordering of `env.path`
                nx.draw_networkx_nodes(nx_graph, pos, nodelist=env.path, ax=graph_ax, node_color=colors, cmap=plt.cm.Blues)  # type: ignore
                nx.draw_networkx_edges(nx_graph, pos, ax=graph_ax) # type: ignore
                nx.draw_networkx_labels(nx_graph, pos, ax=graph_ax) # type: ignore

        if visualize:
            loss_ax.clear()
            length_ax.clear()

            loss_ax.set_title("Loss Plot")
            length_ax.set_title("Length Plot")

            loss_ax.plot(losses)
            length_ax.plot(lengths)
            if notebook:
                display(fig)
                clear_output(wait=True)
            plt.pause(0.000001)

    log_fn(f"Mean path length: {np.mean(lengths)}\tMean length among last 250: {np.mean(lengths[:-250])}\tlast 500: {np.mean(lengths[:-500])}")
    if notebook:
        close_log_file(log_file)  # type: ignore


def create_log_file() -> TextIOWrapper:
    if not os.path.exists("logs"):
        os.mkdir("logs")
    return open(f"logs/log{str(datetime.today().replace(microsecond=0)).replace(' ', '_')}.txt", "w")

def write_log_file(file: TextIOWrapper) -> Callable[[str], None]:
    def write_fn(msg: str) -> None:
        file.write(msg + "\n")
        file.flush()
    return write_fn

def close_log_file(file: TextIOWrapper) -> None:
    file.close()


if __name__ == "__main__":
    train_model(
        visualize=False,
        notebook=False,
        random=False,
        episodes=1,
        num_verts=6,
        num_edges=2,
        delta_e=0,
        prepopulate=True
    )