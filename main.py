import torch

from tqdm import tqdm
from itertools import count
from data import load_corpus
from model import DeepHamModel, DeepHamLoss
from torch.distributions import Categorical
from GraphState import GraphState, Reward

DATA_PATH = './data'
LEARNING_RATE = 0.001
N_EPISODES = 100

# Run one episode
# TODO: TYPES!
def run_episode(model: DeepHamModel, env: GraphState, optimizer: torch.optim.Optimizer, criterion: DeepHamLoss):
    # Clear gradients
    optimizer.zero_grad()

    state: GraphState = env.reset()

    rewards: list[Reward] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    while True:
        probs, value = model(state)  # Perform a single forward pass.
        distribution = Categorical(probs)
        # TODO: this cast should not be necessary
        chosen_action: int = int(distribution.sample().item())

        state, reward, done, _ = env.step(chosen_action)
        
        values.append(value)
        rewards.append(reward)
        log_probs.append(distribution.log_prob(chosen_action))

        if done:
            break

    loss = criterion(log_probs, values, rewards)
    loss.backward()
    return loss


def main():
    corpus = load_corpus(DATA_PATH)

    model = DeepHamModel()
    criterion = DeepHamLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for graph in corpus[0:1]:
        env = GraphState(graph)
        for _ in range(N_EPISODES):
            run_episode(model, env, optimizer, criterion)


if __name__ == "__main__":
    main()
