import os
import random
import time
from collections import OrderedDict, deque

import numpy as np
import torch.nn as nn
import torch
import gym

from common.explorers import LinearExplorer
from common.memory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))

env = gym.make('LunarLander-v2')

if not os.path.exists('models'):
    os.makedirs('models')

MODEL_PATH = 'models/lunar.pt'
PROCESSED_FRAME_SIZE = [84, 84]


class DQN(nn.Module):

    def __init__(self, state_shape, n_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_shape, 200),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        init_f = lambda m: torch.nn.init.xavier_uniform_(m.weight) if type(m) == nn.Linear else None

        self.layers.apply(init_f)
        self.value_stream.apply(init_f)
        self.advantage_stream.apply(init_f)

    def forward(self, state):
        common = self.layers(state)
        value = self.value_stream(common)
        advantage = self.advantage_stream(common)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# training hyperparams
memory_size = 50000
prefill_memory = 10000
batch_size = 32
lr = 0.0001
gamma = 0.99  # Discounting rate
target_net_update_freq = 1000
episodes_train = 1000000
update_frequency = 2


def learn(dqn, target_dqn, memory, criterion, optimizer):
    batch = memory.sample(batch_size)

    rewards = torch.tensor(batch.reward).to(device)
    states = torch.tensor(batch.state).to(device)
    actions = torch.tensor(batch.action).view(-1, 1).to(device)
    next_states = torch.tensor(batch.next_state).to(device)
    dones = torch.tensor(batch.done).to(device)

    with torch.no_grad():
        # double dqn: to get q_values of next_state, get the actions from
        # dqn and use to get qvalues for those actions using target_dqn
        # calculate next actions:
        next_state_actions = dqn(next_states[dones == False]).argmax(dim=1)
        # calculate qvalues using target_dqn
        next_state_qs = (target_dqn(next_states[dones == False])
                         .gather(1, next_state_actions.unsqueeze(1))
                         .squeeze()
                         .to(device))

    q_expected = rewards
    q_expected[dones == False] += gamma * next_state_qs

    # get q values only of played moves
    q_predicted = dqn(states).gather(1, actions).squeeze().to(device)

    loss = criterion(q_predicted, q_expected)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    return loss


def predict_action(dqn, explorer, state, n_actions, steps):
    if False and explorer.should_explore(steps):
        # exploration
        action = random.randint(0, n_actions - 1)
    else:
        # exploitation
        with torch.no_grad():
            qs = dqn(torch.tensor(state).unsqueeze(0).to(device))
        action = torch.argmax(qs).item()

    return action, explorer.explore_prob(steps)


def fill_memory(memory):
    state = env.reset()
    for i in range(prefill_memory):
        action = random.randint(0, env.action_space.n - 1)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        if done:
            state = env.reset()
        else:
            state = next_state


def train():
    memory = ReplayMemory(memory_size)
    fill_memory(memory)
    print('finished filling memory')
    explorer = LinearExplorer(1, 0.1, 100000, 0.01, 1000000)
    dqn = DQN(state_shape=env.observation_space.shape[0],
              n_actions=env.action_space.n).to(device)
    target_dqn = DQN(state_shape=env.observation_space.shape[0],
                     n_actions=env.action_space.n).to(device)

    criterion = torch.nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)

    latest_rewards = deque([], maxlen=100)
    total_steps = 0

    ts_frame = 0
    ts = time.time()

    for episode in range(episodes_train):

        episode_rewards = 0
        losses = []
        state = env.reset()
        done = False

        while not done:
            action, explore_probability = predict_action(dqn,
                                                         explorer,
                                                         state,
                                                         env.action_space.n,
                                                         total_steps)

            next_state, reward, done, _ = env.step(action)

            # env.render()
            episode_rewards += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state

            if total_steps % update_frequency == 0:
                loss = learn(dqn, target_dqn, memory, criterion, optimizer)
                losses.append(loss.item())

            if done:
                speed = (total_steps - ts_frame) / (time.time() - ts)
                ts_frame = total_steps
                ts = time.time()

                latest_rewards.append(episode_rewards)

                print('Episode: {}'.format(episode),
                      'reward: {}'.format(episode_rewards),
                      'explore P: {:.4f}'.format(explore_probability),
                      'loss: {:.4f}'.format(np.mean(losses)),
                      'steps: {}'.format(total_steps),
                      'speed: {:.1f} frames/sec'.format(speed),
                      'average 100: {:.2f}'.format(np.mean(latest_rewards)))

            if total_steps % target_net_update_freq == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            total_steps += 1

        if episode % 10 == 0:
            torch.save(dqn.state_dict(), MODEL_PATH)


def play():
    dqn = DQN(state_shape=env.observation_space.shape[0],
              n_actions=env.action_space.n)

    # dqn.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    for episode in range(500000):
        state = env.reset()
        done = False
        episode_score = 0
        while not done:
            time.sleep(0.01)

            with torch.no_grad():
                qs = dqn(torch.tensor(state).unsqueeze(0))
            action = torch.argmax(qs).item()

            next_state, reward, done, _ = env.step(action)
            episode_score += reward
            env.render()
            state = next_state
        print('episode score {}'.format(episode_score))


def main():
    #train()
    play()
    # display_processd_frame()


if __name__ == '__main__':
    main()
