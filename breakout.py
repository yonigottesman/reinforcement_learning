import os
import random
import time

import numpy as np
import torch.nn as nn
import torch
import gym
import torch.nn.functional as F

from Explorer import Explorer
from LinearExplorer import LinearExplorer
from StackedFrames import StackedFrames
import gym_wrappers
from memory import ReplayMemory, fill_memory
from skimage import transform
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))

env = gym.make('BreakoutNoFrameskip-v4')
env = gym_wrappers.MaxAndSkipEnv(env)
env = gym_wrappers.NoopResetEnv(env)

env = gym_wrappers.EpisodicLifeEnv(env)
env = gym_wrappers.FireResetEnv(env)
env = gym_wrappers.ClipRewardEnv(env)

if not os.path.exists('models'):
    os.makedirs('models')

MODEL_PATH = 'models/breakout.pt'
PROCESSED_FRAME_SIZE = [84, 84]


class DQN(nn.Module):

    # state_shape only hight and width
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        flatten_size = self.calc_flatten_size(state_shape)

        # Dueling network
        self.v_fc = nn.Linear(flatten_size, 512)
        self.v_stream = nn.Linear(512, 1)

        self.advantage_fc = nn.Linear(flatten_size, 512)
        self.advantage_stream = nn.Linear(512, n_actions)

        self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

        torch.nn.init.xavier_uniform_(self.v_fc.weight)
        torch.nn.init.xavier_uniform_(self.v_stream.weight)
        torch.nn.init.xavier_uniform_(self.advantage_fc.weight)
        torch.nn.init.xavier_uniform_(self.advantage_stream.weight)

    def calc_flatten_size(self, state_shape):
        flatten_size = DQN.conv_size_out(np.array(state_shape), self.conv1.kernel_size, self.conv1.stride)
        flatten_size = DQN.conv_size_out(flatten_size, self.conv2.kernel_size, self.conv2.stride)
        flatten_size = DQN.conv_size_out(flatten_size, self.conv3.kernel_size, self.conv3.stride)
        flatten_size = flatten_size.prod() * self.conv3.out_channels
        return flatten_size

    @staticmethod
    def conv_size_out(input_size, kernel_size, stride, padding=0):
        return (input_size + 2 * padding - kernel_size) // stride + 1

    @staticmethod
    def flatten(x):
        return x.view(x.size(0), -1)

    def forward(self, state):
        normalized = state.float() / 255
        a2 = F.relu(self.conv1(normalized))
        a3 = F.relu(self.conv2(a2))
        a4 = F.relu(self.conv3(a3))

        a4_flat = DQN.flatten(a4)

        # Dueling
        a5_v = F.relu(self.v_fc(a4_flat))
        value = self.v_stream(a5_v)

        a5_a = F.relu(self.advantage_fc(a4_flat))
        advantage = self.advantage_stream(a5_a)

        q = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q


def process_frame(frame):
    gray = np.mean(frame, axis=2).astype(np.uint8)
    cropped_frame = gray[25:-12, 4:-4]
    resized = transform.resize(cropped_frame,
                               PROCESSED_FRAME_SIZE,
                               preserve_range=True).astype(np.uint8)
    return resized


# training hyperparams
memory_size = 1000000
prefill_memory = 50000
batch_size = 32
lr = 0.00001
gamma = 0.99  # Discounting rate
target_net_update_freq = 10000
episodes_train = 1000000
update_frequency = 4


def learn(dqn, target_dqn, memory, criterion, optimizer):

    batch = memory.sample(batch_size)

    rewards = torch.tensor(batch.reward).to(device)
    states = torch.cat(batch.state).to(device)
    actions = torch.tensor(batch.action).view(-1, 1).to(device)
    next_states = torch.cat(batch.next_state).to(device)
    dones = torch.tensor(batch.done).to(device)

    with torch.no_grad():
        # double dqn: to get q_values of next_state, get the actions from
        # dqn and use to get qvalues for those actions using target_dqn
        # calculate next actions:
        next_state_actions = dqn(next_states[dones == False]).argmax(dim=1)
        # calculate qvalues using target_dqn
        next_state_qs = target_dqn(next_states[dones == False]).gather(1, next_state_actions.unsqueeze(1)).squeeze().to(
            device)

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
    if explorer.should_explore(steps):
        # exploration
        action = random.randint(0, n_actions - 1)
    else:
        # exploitation
        with torch.no_grad():
            qs = dqn(state.to(device))
        action = torch.argmax(qs).item()

    return action, explorer.explore_prob(steps)


def fill_memory(memory):
    frame_stack = StackedFrames(4, PROCESSED_FRAME_SIZE)

    state = env.reset()
    state = frame_stack.push_get(process_frame(state), True)

    for i in range(prefill_memory):
        action = random.randint(0, env.action_space.n - 1)
        next_state, reward, done, _ = env.step(action)
        next_state = frame_stack.push_get(process_frame(next_state))
        memory.push(state, action, reward, next_state, done)
        if done:
            state = env.reset()
            state = frame_stack.push_get(process_frame(state), True)
        else:
            state = next_state


def train():
    memory = ReplayMemory(memory_size)
    fill_memory(memory)
    print('finished filling memory')
    explorer = LinearExplorer(1, 0.1, 1000000, 0.01, 24000000)
    dqn = DQN(state_shape=PROCESSED_FRAME_SIZE,
              n_actions=env.action_space.n).to(device)
    target_dqn = DQN(state_shape=PROCESSED_FRAME_SIZE,
                     n_actions=env.action_space.n).to(device)

    criterion = torch.nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)

    frame_stack = StackedFrames(4, PROCESSED_FRAME_SIZE)
    rewards_list = []
    total_steps = 0

    ts_frame = 0
    ts = time.time()

    for episode in range(episodes_train):

        episode_rewards = 0
        losses = []
        state = env.reset()
        state = frame_stack.push_get(process_frame(state), True)
        done = False

        while not done:
            action, explore_probability = predict_action(dqn,
                                                         explorer,
                                                         state,
                                                         env.action_space.n,
                                                         total_steps)

            next_state, reward, done, _ = env.step(action)
            next_state = frame_stack.push_get(process_frame(next_state))

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

                rewards_list.append(episode_rewards)

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(episode_rewards),
                      'Explore P: {:.4f}'.format(explore_probability),
                      'Training Loss {}'.format(np.mean(losses)),
                      'total steps {}'.format(total_steps),
                      'speed {} frames/sec'.format(speed))

            if total_steps % target_net_update_freq == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            total_steps += 1

        if episode % 100 == 0:
            torch.save(dqn.state_dict(), MODEL_PATH)


def play():
    dqn = DQN(state_shape=PROCESSED_FRAME_SIZE,
              n_actions=env.action_space.n)

    dqn.load_state_dict(torch.load('models/breakout.pt', map_location=torch.device('cpu')))
    frame_stack = StackedFrames(4, PROCESSED_FRAME_SIZE)
    explorer = Explorer(0, 0, 0)
    for episode in range(500000):
        state = env.reset()
        state = frame_stack.push_get(process_frame(state), True)
        done = False
        episode_score = 0
        while not done:
            time.sleep(0.02)

            with torch.no_grad():
                qs = dqn(state.to(device))
            action = torch.argmax(qs).item()

            next_state, reward, done, _ = env.step(action)
            next_state = frame_stack.push_get(process_frame(next_state))
            episode_score += reward
            env.render()
            state = next_state
        print('episode score {}'.format(episode_score))


def display_processd_frame():
    env.reset()
    observation, reward, done, _ = env.step(env.action_space.sample())
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(observation)
    proc = process_frame(observation)
    ax[1].imshow(proc)
    plt.show()


def main():
    train()
    #play()
    # display_processd_frame()


if __name__ == '__main__':
    main()
