import os
import random

import numpy as np
import torch.nn as nn
import torch
import gym
import torch.nn.functional as F
from common.explorers import ExpExplorer
from common.memory import ReplayMemory, fill_memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1')

if not os.path.exists('models'):
    os.makedirs('models')


MODEL_PATH = 'models/cartpole.pt'


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
    batch = memory.sample(batch_size)

    rewards = torch.tensor(batch.reward).float()
    states = torch.tensor(np.array(batch.state)).float()
    actions = torch.tensor(batch.action).view(-1, 1)
    next_states = torch.tensor(batch.next_state).float()
    dones = torch.tensor(batch.done).float()

    with torch.no_grad():
        next_state_qs = dqn(next_states[dones == False])

    q_expected = rewards
    q_expected[dones == False] += \
        gamma * torch.max(next_state_qs, dim=1).values

    # get q values only of played moves
    q_predicted = dqn(states).gather(1, actions).squeeze()
    loss = criterion(q_predicted, q_expected)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    return loss


def predict_action(dqn, explorer, state, n_actions):
    if explorer.explore():
        # exploration
        action = random.randint(0, n_actions - 1)
    else:
        # exploitation
        with torch.no_grad():
            qs = dqn(torch.from_numpy(state).float())
        action = torch.argmax(qs).item()

    return action, explorer.explore_prob()


def train():
    memory = ReplayMemory(memory_size)
    fill_memory(memory, env, batch_size)
    explorer = ExpExplorer(explore_start, explore_stop, decay_rate)
    dqn = DQN(state_shape=env.observation_space.shape[0],
              n_actions=env.action_space.n).to(device)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)

    rewards_list = []

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

        if episode % 100 == 0:
            torch.save(dqn.state_dict(), MODEL_PATH)

def play():
    dqn = DQN(state_shape=env.observation_space.shape[0],
              n_actions=env.action_space.n)
    # dqn.load_state_dict(torch.load(MODEL_PATH))

    while True:
        state = env.reset()
        done = False
        episode_score = 0
        while not done:
            with torch.no_grad():
                qs = dqn(torch.from_numpy(state).float())
            action = torch.argmax(qs).item()
            next_state, reward, done, _ = env.step(action)
            if done:

                for i in range(100):
                    env.render()
                    env.step(random.randint(0, env.action_space.n - 1))
            episode_score += reward
            env.render()
            state = next_state
        print('episode score {}'.format(episode_score))



def main():
    #train()
    play()


if __name__ == '__main__':
    main()
