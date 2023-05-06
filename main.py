from data import load_corpus
from model import DeepHamModel, DeepHamLoss
from torch.distributions import Categorical
from GraphState import GraphState, Reward
from ReplayBuffer import ReplayBuffer
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric as pyg
import torch
import numpy as np
from IPython.display import display, clear_output
from io import TextIOWrapper
from datetime import datetime
from typing import Callable
import os
torch.manual_seed(0)
np.random.seed(0)


DATA_PATH = './data'
LEARNING_RATE = 0.001
N_EPISODES = 500
PRINT_FREQUENCY = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_episode(model: DeepHamModel, env: GraphState, optimizer: torch.optim.Optimizer, criterion: DeepHamLoss):
    # Clear gradients
    optimizer.zero_grad()

    state: GraphState = env.reset(new_graph=False)

    rewards: list[Reward] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    done = False

    while not done:
        state.graph = state.graph.to(device)
        probs, value = model(state)  # Perform a single forward pass.
        distribution = Categorical(probs.t())
        # Singleton tensor
        chosen_action: torch.Tensor = distribution.sample()

        state, reward, done, _ = env.step(chosen_action.item())  # type: ignore

        values.append(value)
        rewards.append(reward)
        log_probs.append(distribution.log_prob(chosen_action))

    loss = criterion(log_probs, values, rewards)
    loss.backward()
    optimizer.step()
    return loss

def run_random_episode(model, env: GraphState, optimizer, criterion) -> torch.Tensor:
    env = env.reset(new_graph=False)
    done = False
    while not done:
        env.graph = env.graph.to(device)
        options, _, _, _ = pyg.utils.k_hop_subgraph(int(env.curr_vertex_index), 1, env.graph.edge_index)  # type: ignore
        options = options.detach().cpu().numpy()
        options = np.delete(options, np.where(options == int(env.curr_vertex_index)))
        opt = np.random.choice(options).item()
        _, _, done, _ = env.step(opt)
    return torch.tensor(0).to(device)

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

def train_model(visualize=True, notebook=False, random=False):
    # corpus = load_corpus(DATA_PATH)

    if notebook and visualize:
        log_file = create_log_file()
        log_fn = write_log_file(log_file)
    else:
        log_fn = print

    model = DeepHamModel().to(device)
    criterion = DeepHamLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    lengths = []
    losses = []
    model.train()

    log_fn("Starting training..." if not random else "Starting random simulation...")

    fig, [loss_ax, length_ax, graph_ax] = plt.subplots(1, 3)

    # env = GraphState() # also set new_graph=True
    # for i in range(N_EPISODES):
    for i, env in enumerate(ReplayBuffer(N_EPISODES)):
        loss = run_episode(model, env, optimizer, criterion) if not random \
                else run_random_episode(model, env, optimizer, criterion)

        lengths.append(len(env.path))
        losses.append(loss.detach().cpu().numpy())

        if i % PRINT_FREQUENCY == PRINT_FREQUENCY - 1:
            # Print epoch info
            log_fn(f"Epoch {i + 1}: path len = {len(env.path)}\t path = {env.path}")

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
            plt.pause(0.01)
        
    log_fn(f"Mean path length: {np.mean(lengths)}\tMean length among last 250: {np.mean(lengths[:-250])}")
    if notebook and visualize:
        close_log_file(log_file)  # type: ignore

def main():
    train_model(visualize=True)

if __name__ == "__main__":
    main()
    input()
