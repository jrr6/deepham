import gymnasium as gym
from model import DeepHamLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make("CartPole-v1")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(4, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
            nn.Softmax()
        )

        self.critic = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)


def run_episode(model: Model, env: gym.Env, optimizer: torch.optim.Optimizer, criterion: DeepHamLoss):
    state, _ = env.reset()

    rewards = []
    log_probs = []
    values = []
    done = False

    while not done:
        probs, value = model(torch.Tensor(state))
        distribution = torch.distributions.Categorical(probs.t())
        chosen_action = distribution.sample()

        state, reward, done, _, _ = env.step(chosen_action.item())

        values.append(value)
        rewards.append(reward)
        log_probs.append(distribution.log_prob(chosen_action))

    optimizer.zero_grad()
    loss = criterion(log_probs, values, rewards)
    loss.backward()
    optimizer.step()
    return loss, rewards


def main():
    model = Model()
    criterion = DeepHamLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for i in range(400):
        loss, rewards = run_episode(model, env, optimizer, criterion)

        if i % 20 == 0:
            print(f"loss: {loss}, reward: {sum(rewards)}")


if __name__ == "__main__":
    main()
