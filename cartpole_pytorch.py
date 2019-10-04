import random

import numpy as np
import torch.nn as nn
import torch
import gym
import torch.nn.functional as F
from Explorer import Explorer
from memory import ReplayMemory, fill_memory, Experience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1')


class DQN(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_shape, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, n_actions)

    def forward(self, state):
        a2 = F.relu(self.fc1(state))
        a3 = F.relu(self.fc2(a2))
        a4 = self.fc3(a3)
        return a4


# training hyperparams
memory_size = 20000
batch_size = 64
lr = 0.0001
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob
gamma = 0.9  # Discounting rate


def learn(dqn, memory, criterion, optimizer):
    n_actions = env.action_space.n
    possible_actions = np.identity(env.action_space.n, dtype=int)

    batch = memory.sample(batch_size)
    batch = Experience(*zip(*batch))
    rewards_mb = torch.from_numpy(np.array(batch.reward)).float()
    states_mb = torch.tensor(np.array(batch.state), requires_grad=True).float()
    actions_mb = torch.tensor(np.array([possible_actions[i] for i in batch.action])).float()

    next_states_mb = torch.from_numpy(np.array(batch.next_state)).float()
    dones_mb = torch.from_numpy(np.array(batch.done)).float()

    with torch.no_grad():
        next_state_qs = dqn(next_states_mb)

    target_qs_batch = rewards_mb
    target_qs_batch[dones_mb == False] += \
        gamma * torch.max(next_state_qs[dones_mb == False], axis=1).values

    # get list of q values for state and chosen action
    y_hat = torch.sum(torch.mul(dqn(states_mb), actions_mb), dim=1)
    loss = criterion(y_hat, target_qs_batch)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    return loss


def predict_action(dqn, explorer, state, n_actions):

    if explorer.explore():
        # exploration
        action = random.randint(0, n_actions-1)
    else:
        # exploitation
        with torch.no_grad():
            qs = dqn(torch.from_numpy(state).float())
        action = torch.argmax(qs).item()

    return action, explorer.explore_prob()


def train():
    memory = ReplayMemory(memory_size)
    fill_memory(memory, env, batch_size)
    explorer = Explorer(explore_start, explore_stop, decay_rate)
    dqn = DQN(state_shape=env.observation_space.shape[0],
              n_actions=env.action_space.n).to(device)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)

    rewards_list = []
    loss = 1
    for episode in range(5000):
        episode_rewards = 0
        state = env.reset()
        done = False
        while not done:
            action, explore_probability = predict_action(dqn,
                                                         explorer,
                                                         state,
                                                         env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            # env.render()
            episode_rewards += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state
            loss = learn(dqn, memory, criterion, optimizer)

            if done:
                rewards_list.append(episode_rewards)
                moving_average = np.mean(rewards_list[-100:])
                if episode % 50 == 0:
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(episode_rewards),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {}'.format(loss),
                          'Moving average {}'.format(moving_average))

        if episode % 10 == 0:
            pass


def main():
    train()


if __name__ == '__main__':
    main()
