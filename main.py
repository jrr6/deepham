from argparse import ArgumentParser, BooleanOptionalAction
from ReplayBuffer import ReplayBuffer
from GraphState import GraphState, Reward
from torch.distributions import Categorical
from model import DeepHamModel, DeepHamLoss, DeepHamActor, DeepHamSupervised
from data import generate_semirandom_hampath_graph

import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric as pyg
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


DATA_PATH = "./data"
LEARNING_RATE = 0.001
N_EPISODES = 500
N_SUPERVISED_EPOCHS = 500
BATCH_SIZE = 20
PRINT_FREQUENCY = 10


def run_episode(
    model: DeepHamModel,
    env: GraphState,
    optimizer: torch.optim.Optimizer,
    criterion: DeepHamLoss,
):
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

    # Clear gradients
    optimizer.zero_grad()
    loss = criterion(log_probs, values, rewards)
    loss.backward()
    optimizer.step()
    return loss


def reinforcement_learning():
    # corpus = load_corpus(DATA_PATH)

    model = DeepHamModel()
    criterion = DeepHamLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    lengths = []
    losses = []
    model.train()

    fig, [loss_ax, length_ax, graph_ax] = plt.subplots(1, 3)
    for i, env in enumerate(ReplayBuffer(N_EPISODES)):
        loss = run_episode(model, env, optimizer, criterion)

        lengths.append(len(env.path))
        losses.append(loss.detach().numpy())

        loss_ax.clear()
        length_ax.clear()

        loss_ax.set_title("Loss Plot")
        length_ax.set_title("Length Plot")

        if i % PRINT_FREQUENCY == PRINT_FREQUENCY - 1:
            graph_ax.clear()
            graph_ax.set_title(f"{env.path}")
            print(
                f"Epoch {i + 1}: loss = {loss.item()}\t path len = {len(env.path)}\t path = {env.path}"
            )
            nx_graph = pyg.utils.to_networkx(env.initial_graph, to_undirected=True)  # type: ignore
            color_offset = 50
            colors = [
                color_offset + env.path.index(v) if v in env.path else 0
                for v in list(nx_graph.nodes)
            ]
            nx.draw_kamada_kawai(  # type: ignore
                nx_graph,
                with_labels=True,
                node_color=colors,
                ax=graph_ax,
                cmap=plt.cm.Blues,  # type: ignore
            )

            plt.pause(0.01)


def run_batch(
    model: DeepHamSupervised,
    batch: np.ndarray,
    criterion: torch.nn.NLLLoss,
    optimizer: torch.optim.Adam,
    training=False,
) -> tuple[float, float]:
    correct = 0
    labels = []
    outputs = []
    for graph, label in batch:
        probs = model(graph)
        chosen_vertex = torch.argmax(probs).item()

        correct += int(chosen_vertex == label)
        labels.append(int(label))
        outputs.append(probs)

    loss = criterion(
        torch.log(torch.stack(outputs)), torch.Tensor(labels).reshape(-1, 1).long()
    )

    acc = correct / len(batch)
    if training:
        optimizer.zero_grad()
        loss.backward()

    return loss, acc


def supervised_learning():
    model = DeepHamSupervised()  # TODO: update this to not have mask
    criterion = torch.nn.NLLLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    corpus = np.array(
        [generate_semirandom_hampath_graph(30, 0, 15, 10) for _ in range(100)],
        dtype=object,
    )
    test_batch = np.array(
        [generate_semirandom_hampath_graph(30, 0, 15, 10) for _ in range(100)],
        dtype=object,
    )

    training_loss = []
    test_loss = []
    training_acc = []
    test_acc = []

    fig, [loss_ax, acc_ax] = plt.subplots(1, 2)  # TODO: pretty plots
    for epoch in range(N_SUPERVISED_EPOCHS):
        np.random.shuffle(corpus)

        model.train(True)
        for batch_idx in range(len(corpus) // BATCH_SIZE):
            batch = corpus[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            loss, acc = run_batch(model, batch, criterion, optimizer, training=True)

            print(
                f"[Train Epoch: {epoch}/{N_SUPERVISED_EPOCHS} Batch: {batch_idx + 1} / {len(corpus) // BATCH_SIZE}] loss: {loss} acc: {acc}"
            )

            training_loss.append(loss)
            training_acc.append(acc)

        model.train(False)
        loss, acc = run_batch(model, test_batch, criterion, optimizer)
        test_loss.append(loss)
        test_acc.append(acc)


if __name__ == "__main__":
    parser = ArgumentParser("DeepHam")
    parser.add_argument("--supervised", action=BooleanOptionalAction)
    parser.add_argument("--reinforcement", action=BooleanOptionalAction)

    args = parser.parse_args()

    supervised_learning()

    if args.reinforcement:
        reinforcement_learning()
        plt.show()
    elif args.supervised:
        supervised_learning()
        plt.show()
    else:
        parser.print_help()
