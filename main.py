import torch
import torch_geometric as pyg
import networkx as nx
import matplotlib

from data import load_corpus
from model import DeepHamModel, DeepHamLoss
from torch.distributions import Categorical
from GraphState import GraphState, Reward

DATA_PATH = './data'
LEARNING_RATE = 0.001
N_EPISODES = 500
PRINT_FREQUENCY = 10

# Run one episode
# TODO: TYPES!


def run_episode(model: DeepHamModel, env: GraphState, optimizer: torch.optim.Optimizer, criterion: DeepHamLoss):
    # Clear gradients
    optimizer.zero_grad()

    state: GraphState = env.reset(new_graph=False)

    rewards: list[Reward] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    done = False

    while not done:
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


def main():
    # corpus = load_corpus(DATA_PATH)

    model = DeepHamModel()
    criterion = DeepHamLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # for graph in corpus[0:1]:
    env = GraphState(None)
    graph = env.graph.clone()  # type: ignore
    model.train()

    for i in range(N_EPISODES):
        loss = run_episode(model, env, optimizer, criterion)

        # nx.draw_kamada_kawai(pyg.utils.to_networkx(graph, to_undirected=True), with_labels=True) # type: ignore

        if i % PRINT_FREQUENCY == PRINT_FREQUENCY - 1:
            print(f"Epoch {i + 1}: loss = {loss.item()}\t path len = {len(env.path)}\t path = {env.path}")
    nx.draw_kamada_kawai(pyg.utils.to_networkx(graph, to_undirected=True), with_labels=True) # type: ignore
    matplotlib.pyplot.show()  # type: ignore


if __name__ == "__main__":
    main()
